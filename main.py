import pandas as pd
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- Security & API Setup ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API Key not found. Please ensure your .env file contains GEMINI_API_KEY.")

genai.configure(api_key=api_key)
# Zero temperature ensures consistent, reproducible results and prevents hallucinations
model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"temperature": 0.0})

# --- 1. Load Data ---
shipments = pd.read_csv('shipment_records.csv')
notes = pd.read_csv('context_notes.csv')

# --- 2. Process Core Metrics ---
shipments['shipment_date'] = pd.to_datetime(shipments['shipment_date'])
shipments['route'] = shipments['origin'] + '-' + shipments['destination']

# Group shipments into Monday-starting weeks
shipments['week_of'] = shipments['shipment_date'] - pd.to_timedelta(shipments['shipment_date'].dt.dayofweek, unit='D')
shipments['tonne_km'] = shipments['quantity_tonnes'] * shipments['distance_km']

# --- 3. Weekly Aggregation ---
weekly = shipments.groupby(['route', 'route_type', 'week_of']).agg(
    total_freight_cost=('freight_cost_inr', 'sum'),
    total_tonne_km=('tonne_km', 'sum')
).reset_index()

# cost per tonne per km = total freight cost / (quantity * distance)
weekly['cost_per_tonne_km'] = weekly['total_freight_cost'] / weekly['total_tonne_km']
weekly = weekly.sort_values(['route', 'week_of']).reset_index(drop=True)

# --- 4. Baseline 1: Trailing 8-Week Average (No Look-Ahead) ---
def calc_trailing(df):
    # shift(1) strictly prevents data leakage from the current week
    df['past_avg'] = df['cost_per_tonne_km'].shift(1).rolling(window=8, min_periods=1).mean()
    return df

weekly = weekly.groupby('route', group_keys=False).apply(calc_trailing)

# --- 5. Baseline 2: Peer Routes (Same Week, Same Type, Exclude Self) ---
peer_stats = weekly.groupby(['week_of', 'route_type'])['cost_per_tonne_km'].agg(
    sum_cost='sum', count_routes='count'
).reset_index()

weekly = weekly.merge(peer_stats, on=['week_of', 'route_type'])
# Algebraic subtraction to calculate average of OTHER routes, excluding the current route
weekly['peer_avg'] = (weekly['sum_cost'] - weekly['cost_per_tonne_km']) / (weekly['count_routes'] - 1)
weekly['peer_avg'] = np.where(weekly['count_routes'] > 1, weekly['peer_avg'], np.nan)

weekly['pct_vs_history'] = ((weekly['cost_per_tonne_km'] - weekly['past_avg']) / weekly['past_avg']) * 100
weekly['pct_vs_peers'] = ((weekly['cost_per_tonne_km'] - weekly['peer_avg']) / weekly['peer_avg']) * 100

# --- 6. Flagging Logic ---
# Flag if cost increased >5% vs history and is >5% higher than peer average
weekly['needs_review'] = (weekly['pct_vs_history'] > 5.0) & (weekly['pct_vs_peers'] > 5.0)

# --- 7. AI Retrieval System (RAG) ---
def check_context_with_ai(row, notes_df):
    if not row['needs_review']:
        return "No", "", "Cost metrics are stable and within normal baseline variance."

    # Prevent hallucination by only passing geographically relevant notes to the context
    applicable_notes = notes_df[notes_df['applies_to'].isin([row['route'], 'All Routes'])]
    
    prompt = f"""
    Route: {row['route']}
    Date: {row['week_of'].strftime('%Y-%m-%d')}
    
    Review these context notes:
    {applicable_notes.to_string()}
    
    Is there a note that explicitly justifies a cost increase for this EXACT route and time period?
    Ignore notes that say "costs were not affected" or "no major disruptions".
    Respond STRICTLY in this format:
    JUSTIFIED: [Yes or No]
    NOTE_ID: [note_id or blank]
    REASON: [1-2 sentences explaining why, or why it is unexplained]
    """
    
    try:
        response = model.generate_content(prompt).text
        lines = response.strip().split('\n')
        is_justified = "Yes" in lines[0]
        note_id = lines[1].split(':')[1].strip() if len(lines) > 1 else ""
        reason = lines[2].split(':')[1].strip() if len(lines) > 2 else "Unexplained spike."
        
        if is_justified and note_id:
            return "No (justified)", note_id, reason
        else:
            return "Yes", "", "No matching note found for this route or date range. Cost rise looks unexplained."
    except Exception as e:
        return "Yes", "", "Error processing note retrieval."

# Apply AI to flagged routes
results = weekly.apply(lambda row: check_context_with_ai(row, notes), axis=1)
weekly[['flagged', 'matched_note_id', 'reason']] = pd.DataFrame(results.tolist(), index=weekly.index)

# --- 8. Formatting output to exact required specification ---
def format_pct(val, context):
    if pd.isna(val): return "N/A"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.1f}% vs {context}"

weekly['vs_own_history'] = weekly['pct_vs_history'].apply(lambda x: format_pct(x, "this route's past average"))
weekly['vs_similar_routes'] = weekly['pct_vs_peers'].apply(lambda x: format_pct(x, "similar-length routes this week"))

final_output = weekly[['route', 'week_of', 'cost_per_tonne_km', 'vs_own_history', 'vs_similar_routes', 'flagged', 'matched_note_id', 'reason']]
final_output['week_of'] = final_output['week_of'].dt.strftime('%Y-%m-%d')
final_output['cost_per_tonne_km'] = final_output['cost_per_tonne_km'].round(2)

final_output.to_csv('submission_output.csv', index=False)
