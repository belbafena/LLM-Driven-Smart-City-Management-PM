import streamlit as st
import pandas as pd
import time
import datetime
from fpdf import FPDF
import base64
import os
import plotly.express as px
import io 
import json
import plotly.express as px
import numpy as np
from rag_module import retrieve_context
from llm_module import generate_decision
import plotly.graph_objects as go

@st.cache_data
def load_data():
    return pd.read_csv("scored_data.csv")

# df = load_data()

# To check current Threshold references
def interpret(value, p50, p75):
    if value >= p75:
        return "High"
    elif value >= p50:
        return "Moderate"
    else:
        return "Low"
    
HISTORY_FILE = "decision_history.json"

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []
# Page config
# st.set_page_config(page_title="Smart City Decision Support", layout="wide")

# st.title("🏙️ Smart City Decision Support System")
# st.markdown("### LLM-Driven Integrated Urban Management")
# Define the text variables (or hardcode them)
# Force Streamlit's main block to be truly wide
# --- PAGE CONFIG ---
st.set_page_config(page_title="Smart City Decision Support", layout="wide")
# --- CSS OVERRIDE FOR FULL WIDTH ---
# --- CSS OVERRIDE FOR FULL WIDTH + CROSS-BROWSER FIX ---
st.markdown("""
    <style>

    /* ✅ KEEP YOUR ORIGINAL */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1400px !important;
        margin: auto;
    }

    /* 🔥 ADD THIS BELOW (DO NOT REMOVE ABOVE) */

    /* Fix layout consistency across Chrome & Edge */
    * {
        box-sizing: border-box;
    }

    /* Fix sidebar width (Edge issue) */
    section[data-testid="stSidebar"] {
        min-width: 260px !important;
        max-width: 260px !important;
    }

    /* Fix column spacing */
    div[data-testid="column"] {
        padding: 0.25rem !important;
    }

    /* Fix button size + prevent text breaking */
    button {
        height: 42px !important;
        font-size: 14px !important;
        border-radius: 8px !important;
        white-space: nowrap !important;
    }

    /* Improve font rendering */
    body {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    </style>
    """, unsafe_allow_html=True)
st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1400px !important;
        margin: auto;
    }
    /* ADD BELOW — do not touch header */

* {
    box-sizing: border-box;
}

section[data-testid="stSidebar"] {
    min-width: 260px !important;
    max-width: 260px !important;
}

div[data-testid="column"] {
    padding: 0.25rem !important;
}

button {
    height: 42px !important;
    font-size: 14px !important;
    border-radius: 8px !important;
    white-space: nowrap !important;
}
    </style>
    """, unsafe_allow_html=True)

# --- LOGO LOADING ---
logo_path = "upb_logo.png" 
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        encoded_logo = base64.b64encode(f.read()).decode()
        logo_html = f"data:image/png;base64,{encoded_logo}"
else:
    logo_html = "" 

# --- HEADER DATA ---
title_text = "Smart City Decision Support System"
subtitle_text = "LLM-Driven Integrated Urban Management"
location_text = "SMART CITY ENVIRONMENT"

# --- RENDER HEADER ---
st.markdown(f"""
    <div style="
        background-color: #f0f2f6;
        padding: 20px 30px;
        border-radius: 12px;
        border-left: 12px solid #007bff;
        margin-bottom: 30px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    ">
        <div style="display: flex; gap: 20px; align-items: center; flex-wrap: nowrap;">
            {"<img src='" + logo_html + "' width='85' style='margin-right: 25px;'>" if logo_html else ""}
            <div>
                <h1 style="margin:0; padding:0; font-size: 2.2rem; color: #1e1e1e;">{title_text}</h1>
                <p style="margin:5px 0 0 0; font-size: 1.1rem; color: #555; font-weight: 500;">{subtitle_text}</p>
            </div>
        </div>
        <div style="flex: 1; text-align: right; border-left: 2px solid #ddd; padding-left: 25px;">
            <p style="margin:0; font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1px;">SYSTEM Context</p>
            <p style="margin:0; color: #007bff; font-size: 1.2rem; font-weight: bold;">{location_text}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
# Load data
df = pd.read_csv("scored_data.csv")

p75_moving = df["MovingAvg3hr"].quantile(0.75)
p50_moving = df["MovingAvg3hr"].quantile(0.5)

p75_lag = df["LagTraffic"].quantile(0.75)
p50_lag = df["LagTraffic"].quantile(0.5)
# ── Move junction_map here ──
junction_map = {
    1: "Highway Junction",
    2: "City Center Intersection",
    3: "Commercial Area",
    4: "Residential Area"
}

# ===============================
# ⚙️ SCENARIO NAVIGATION + FILTER (FINAL FIX)
# ===============================
st.sidebar.header("⚙️ Navigation")

# Initialize session state
if "row_index" not in st.session_state:
    st.session_state.row_index = 0

# 🔍 Filter
st.sidebar.markdown("### 🔍 Scenario Filter")
filter_high = st.sidebar.checkbox("Show Only HIGH Congestion")

# Reset index when filter changes
if "prev_filter" not in st.session_state:
    st.session_state.prev_filter = filter_high

if filter_high != st.session_state.prev_filter:
    st.session_state.row_index = 0
    st.session_state.prev_filter = filter_high

# Apply filter FIRST
if filter_high:
    filtered_df = df[df["I_Congestion"] == 1].reset_index(drop=True)
else:
    filtered_df = df.reset_index(drop=True)
st.markdown("""
<style>
button {
    white-space: nowrap !important;
    font-size: 14px !important;
}
</style>
""", unsafe_allow_html=True)
# Navigation buttons
col1, col2 = st.sidebar.columns([1,1])

if col1.button("⬅ Previous"):
    if st.session_state.row_index > 0:
        st.session_state.row_index -= 1
        st.session_state.decision = None   # 🔥 reset decision

if col2.button("Next ➡"):
    if st.session_state.row_index < len(filtered_df) - 1:
        st.session_state.row_index += 1
        st.session_state.decision = None   # 🔥 reset decision

# Clamp index (VERY IMPORTANT)
if st.session_state.row_index < 0:
    st.session_state.row_index = 0

if st.session_state.row_index >= len(filtered_df):
    st.session_state.row_index = len(filtered_df) - 1

# Final row selection
row = filtered_df.iloc[st.session_state.row_index]

# Display info
st.sidebar.write(f"Scenario: {st.session_state.row_index + 1} / {len(filtered_df)}")

# Priority badge in sidebar
priority = "CRITICAL" if row["P_Congestion1"] >= 0.85 else \
           "HIGH"     if row["P_Congestion1"] >= 0.60 else \
           "MODERATE" if row["P_Congestion1"] >= 0.40 else "NORMAL"

priority_color = {
    "CRITICAL": "#dc3545",
    "HIGH":     "#fd7e14",
    "MODERATE": "#ffc107",
    "NORMAL":   "#28a745"
}[priority]

priority_emoji = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MODERATE": "🟡",
    "NORMAL":   "🟢"
}[priority]

st.sidebar.markdown(f"""
<div style="
    background:{priority_color}22;
    border:1.5px solid {priority_color};
    border-radius:8px;
    padding:8px 12px;
    text-align:center;
    margin-top:6px;
">
    <div style="font-size:0.75rem;color:#666;">SAS Priority</div>
    <div style="font-size:1.1rem;font-weight:bold;color:{priority_color};">
        {priority_emoji} {priority}
    </div>
    <div style="font-size:0.75rem;color:#888;">
        P_Congestion1 = {row['P_Congestion1']:.3f}
    </div>
</div>
""", unsafe_allow_html=True)

# DEBUG (remove later)
# st.sidebar.write("Current value:", row["I_Congestion"])

# Progress bar (nice UX)
st.sidebar.progress((st.session_state.row_index + 1) / len(filtered_df))
auto_play = st.sidebar.checkbox("▶️ Auto Play Simulation")

if auto_play:
    time.sleep(1)
    st.session_state.row_index = (st.session_state.row_index + 1) % len(filtered_df)
    st.rerun()

# Select row
row = filtered_df.iloc[st.session_state.row_index]
# ── Initialize session state keys ──────────────────
if "decision" not in st.session_state:
    st.session_state.decision = None

if "decision_history" not in st.session_state:
    st.session_state.decision_history = []
# ── Define early so sidebar decision history can use them ──
junction_name_early = junction_map.get(int(row["Junction"]), "Unknown") \
    if hasattr(row, "Junction") else "Unknown"
congestion_early = "HIGH CONGESTION" if int(row["I_Congestion"]) == 1 else "NORMAL TRAFFIC"
confidence_early = row["P_Congestion1"] if int(row["I_Congestion"]) == 1 else row["P_Congestion0"]
input_time_early = f"{int(row['Hour']):02d}:00"
# ===============================
# 📋 DECISION HISTORY LOG
# ===============================
if "decision_history" not in st.session_state:
    st.session_state.decision_history = load_history()  # ← loads from file

# if st.session_state.decision:
#     current_entry = {
#         "Scenario":   st.session_state.row_index + 1,
#         "Junction":   junction_name_early,
#         "Hour":       input_time_early,
#         "Status":     congestion_early,
#         "Confidence": f"{confidence_early:.3f}",
#         "Timestamp":  datetime.datetime.now().strftime("%H:%M:%S")
#     }
#     existing = [h["Scenario"] for h in st.session_state.decision_history]
#     if st.session_state.row_index + 1 not in existing:
#         st.session_state.decision_history.append(current_entry)
#         save_history(st.session_state.decision_history)  # ← save to file

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Decision Log")

if st.session_state.decision_history:
    for entry in reversed(st.session_state.decision_history[-5:]):
        color = "#dc3545" if "HIGH" in entry["Status"] else "#28a745"
        st.sidebar.markdown(f"""
        <div style="
            background:{color}11;
            border-left:3px solid {color};
            border-radius:4px;
            padding:6px 8px;
            margin-bottom:6px;
            font-size:0.75rem;
        ">
            <b>#{entry['Scenario']} {entry['Junction']}</b><br>
            {entry['Hour']} · {entry['Status'].replace('HIGH CONGESTION','🔴 HIGH').replace('NORMAL TRAFFIC','🟢 NORMAL')}<br>
            Conf: {entry['Confidence']} · {entry['Timestamp']}
        </div>
        """, unsafe_allow_html=True)

    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.decision_history = []
        save_history([])        # ← clears file too
        st.rerun()
else:
    st.sidebar.caption("No decisions generated yet.")

# ===============================
# 🧠 PREPARE INPUT
# ===============================
congestion = "HIGH CONGESTION" if int(row["I_Congestion"]) == 1 else "NORMAL TRAFFIC"

# Handle LagTraffic properly
lag_traffic = row["LagTraffic"]

if pd.isna(lag_traffic):
    lag_display = "N/A"
    lag_value = 0
else:
    lag_display = round(lag_traffic, 2)
    lag_value = lag_traffic

input_data = {
    "congestion": congestion,
    "moving_avg": row["MovingAvg3hr"],
    "lag_traffic": lag_value,
    "rainfall": row["RAIN_MM"],
    "time": f"{int(row['Hour']):02d}:00"
}

if int(row["I_Congestion"]) == 1:
    confidence = row["P_Congestion1"]   # confidence of HIGH congestion
else:
    confidence = row["P_Congestion0"]   # confidence of NORMAL traffic

# ===============================
# 🎨 COLOR LOGIC (FIXED)
# ===============================
def get_color(level):
    if "HIGH CONGESTION" in level:
        return "red"
    elif "NORMAL TRAFFIC" in level:
        return "green"
    else:
        return "orange"

color = get_color(congestion)

st.subheader("🚦 Simulated Real-Time Traffic Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

def metric_card(title, value, color="#1e1e1e"):
    return f"""
    <div style="
        background-color: white;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 3px 6px rgba(0,0,0,0.08);
        text-align: center;
        height: 100%;
        height: 95px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        white-space: nowrap; 
    ">
        <div style="font-size: 0.85rem; color: #777;">{title}</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: {color};">
            {value}
        </div>
    </div>
    """

# Dynamic color for congestion
congestion_color = "#28a745" if "NORMAL" in congestion else "#dc3545"

with col1:
    st.markdown(metric_card("Congestion", congestion, congestion_color), unsafe_allow_html=True)

with col2:
    st.markdown(metric_card("Moving Avg", round(row["MovingAvg3hr"], 2)), unsafe_allow_html=True)

with col3:
    st.markdown(metric_card("Lag Traffic", lag_display), unsafe_allow_html=True)

with col4:
    st.markdown(metric_card("Rainfall", row["RAIN_MM"]), unsafe_allow_html=True)

with col5:
    st.markdown(metric_card("Time", input_data["time"]), unsafe_allow_html=True)
# Visual status box
# st.markdown(f"""
# <div style="background-color:{color};padding:12px;border-radius:10px;color:white;text-align:center;">
# <h3>Traffic Status: {congestion}</h3>
# </div>
# """, unsafe_allow_html=True)
def status_bar(congestion):

    if "HIGH" in congestion:
        bg = "linear-gradient(90deg, #ff5c5c, #d60000)"
        icon = "⚠️"
        text = "HIGH CONGESTION"
    else:
        bg = "linear-gradient(90deg, #28a745, #1e7e34)"
        icon = "🚦"
        text = "NORMAL TRAFFIC"

    return f"""
    <div style="
        background: {bg};
        padding: 14px 18px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-weight: 600;
        margin-top: 10px; 
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        letter-spacing: 0.5px;
    ">
        <div style="font-size: 0.9rem; opacity: 0.9;">
            CURRENT TRAFFIC STATUS
        </div>
        <div style="font-size: 1.4rem; margin-top: 5px;">
            {icon} {text}
        </div>
    </div>
    """

st.markdown(status_bar(congestion), unsafe_allow_html=True)
# junction_map = {
#     1: "Highway Junction",
#     2: "City Center Intersection",
#     3: "Commercial Area",
#     4: "Residential Area"
# }

junction_name = junction_map.get(int(row["Junction"]), "Unknown")

st.markdown(f"""
<div style="padding:5px;">
📍 <b>Location:</b> {junction_name}
</div>
""", unsafe_allow_html=True)


col_left, col_right = st.columns([3, 1])

with col_left:
    st.subheader("📈 Traffic Trends")

    # Select window around current scenario
    start = max(0, st.session_state.row_index - 20)
    end = st.session_state.row_index + 1

    chart_df = filtered_df.iloc[start:end][["Hour", "MovingAvg3hr", "LagTraffic"]].fillna(0)

    # Create figure
    fig = go.Figure()

    # 1. Add Threshold Lines (Horizontal)
    # These provide the "Statistical Context"
    fig.add_hline(
        y=p75_moving, 
        line_dash="dot", 
        line_color="orange", 
        annotation_text="P75 (Critical)", 
        annotation_position="top left"
    )
    fig.add_hline(
        y=p50_moving, 
        line_dash="dot", 
        line_color="gray", 
        annotation_text="P50 (Median)", 
        annotation_position="bottom left"
    )

    # 2. Moving Average Trace
    fig.add_trace(go.Scatter(
        x=chart_df.index,
        y=chart_df["MovingAvg3hr"],
        mode='lines',
        name='Moving Avg',
        line=dict(width=3),
        customdata=chart_df["Hour"],
        hovertemplate='Time: %{customdata}:00<br>Moving Avg: %{y:.2f}<extra></extra>'
    )) 

    # 3. Lag Traffic Trace
    fig.add_trace(go.Scatter(
        x=chart_df.index,
        y=chart_df["LagTraffic"],
        mode='lines',
        name='Lag Traffic',
        customdata=chart_df["Hour"],
        hovertemplate='Time: %{customdata}:00<br>Lag Traffic: %{y}<extra></extra>'
    ))

    # 4. 🔴 Current point (vertical line)
    fig.add_vline(
        x=chart_df.index[-1],
        line_width=3,
        line_dash="dash",
        line_color="red"
    )
    
    # Optional: Add a "LIVE" label to the vertical line
    fig.add_annotation(
        x=chart_df.index[-1],
        y=1.05,
        yref="paper",
        text="LIVE",
        showarrow=False,
        font=dict(color="red", size=10, family="Arial Black")
    )

    # Layout
    fig.update_layout(
        height=300, # Increased slightly to fit annotations
        margin=dict(l=10, r=10, t=50, b=40),
        legend=dict(
            orientation="h",
            y=1.2,
            x=0
        ),
        yaxis=dict(title="Traffic Volume")
    )

    st.plotly_chart(fig, width="stretch")

with col_right:
    st.subheader("📊 Model Confidence")
    if confidence > 0.8:
        st.success(f"Confidence: {confidence:.2f} — High")
    elif confidence > 0.6:
        st.warning(f"Confidence: {confidence:.2f} - Moderate")
    else:
        st.error(f"Confidence: {confidence:.2f} - Low (Uncertain)")
    st.markdown("---")

    st.subheader("📈 Threshold Reference")

    st.write("**Moving Average (3hr)**")
    st.write(f"P50: {p50_moving:.2f}")
    st.write(f"P75: {p75_moving:.2f}")

    st.write("**Lag Traffic**")
    st.write(f"P50: {p50_lag:.2f}")
    st.write(f"P75: {p75_lag:.2f}")
    st.markdown("---")
    
    st.subheader("📊 Current Level")

    st.write(
        f"Moving Avg Level: {interpret(row['MovingAvg3hr'], p50_moving, p75_moving)}"
    )
    st.write(
        f"Lag Traffic Level: {interpret(row['LagTraffic'], p50_lag, p75_lag)}"
    )
# ===============================
# 📚 RAG CONTEXT
# ===============================
context = retrieve_context(input_data)

col1, col2 = st.columns(2)

# with col1:
#     st.subheader("📚 Context Insights (RAG Reasoning)")
#     if context:
#         for c in context:
#             st.markdown(f"- {c}")
#     else:
#         st.info("No Contextual Insights available for this Scenario.")
with col1:
    st.subheader("📚 Contextual Evidence")
    # Wrap the RAG output in an expander
    with st.expander("📖 View Retrieved City Policies & Cases", expanded=False):
        if context:
            for c in context:
                st.info(c) # Using st.info adds a nice blue background to each rule
        else:
            st.write("No specific historical data found.")

# with col2:
#     st.subheader("🧠 LLM Decision")

#     if "decision" not in st.session_state:
#         st.session_state.decision = None

#     if st.button("🚀 Generate Decision"):
#         with st.spinner("Analyzing..."):
#             st.session_state.decision = generate_decision(input_data, context, confidence)

#     if st.session_state.decision:
#         st.markdown(st.session_state.decision)
with col2:
    st.subheader("🧠 Decision Engine")
    if "decision" not in st.session_state:
       st.session_state.decision = None
    if st.button("🚀 Generate Management Decision"):
        # Use a status box to show the "Thinking" process
        with st.status("🤖 AI analyzing data...", expanded=True) as status:
            st.write("Fetching SAS Model Confidence...")
            time.sleep(0.5)
            st.write("Applying RAG Urban Policies...")
            time.sleep(0.5)
            
            # Call your LLM function
            st.session_state.decision = generate_decision(input_data, context, confidence)
            
            status.update(label="✅ Analysis Complete", state="complete", expanded=False)

    # Display the final decision outside the status box
    if st.session_state.decision:
        st.markdown(st.session_state.decision)
# ← ADD THIS BLOCK immediately after button
    current_entry = {
        "Scenario":   st.session_state.row_index + 1,
        "Junction":   junction_name,
        "Hour":       input_data["time"],
        "Status":     congestion,
        "Confidence": f"{confidence:.3f}",
        "Timestamp":  datetime.datetime.now().strftime("%H:%M:%S")
    }
    existing = [h["Scenario"] for h in st.session_state.decision_history]
    if st.session_state.row_index + 1 not in existing:
        st.session_state.decision_history.append(current_entry)
        save_history(st.session_state.decision_history)
    # st.rerun()  # ← forces sidebar to refresh and show new entry
def generate_pdf_report(junction, time, input_data, decision, is_high, context_sources):
    pdf = FPDF()
    pdf.add_page()
    
    # 1. HEADER (Color coded)
    if is_high:
        pdf.set_fill_color(255, 200, 200)
        title = f"ALARM: HIGH CONGESTION - {junction}"
    else:
        pdf.set_fill_color(200, 255, 200)
        title = f"STATUS: NORMAL - {junction}"
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 15, title, ln=True, align='C', fill=True)
    pdf.ln(5)

    # 2. METADATA
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 8, f"Scenario Time: {time}", ln=True)
    pdf.ln(5)

    # 3. SAS DATA SUMMARY
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "SAS Predictive Metrics", ln=True)
    pdf.set_font("Arial", size=11)
    stats = (f"- State: {input_data['congestion']}\n"
             f"- Moving Avg: {input_data['moving_avg']:.2f}\n"
             f"- Lag Traffic: {input_data['lag_traffic']:.2f}\n"
             f"- Rainfall: {input_data['rainfall']} mm")
    pdf.multi_cell(0, 8, stats)
    pdf.ln(5)

    # Thresholds
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, "Statistical Benchmarks (Thresholds)", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 7, f"- P50 (Median): {p50_moving:.2f}\n- P75 (Critical): {p75_moving:.2f}")
    pdf.ln(5)

    # 4. HISTORICAL REASONING SOURCES — properly labeled by type
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Historical Reasoning Sources (CBR Layer)", ln=True)

    if context_sources:
        for source in context_sources:
            # Determine label based on content
            if "Case ID" in source:
                label = "Case Match"
                pdf.set_text_color(0, 100, 180)       # blue for case
            elif "Recommended strategy" in source:
                label = "Strategy"
                pdf.set_text_color(0, 130, 80)         # green for strategy
            elif "intensity" in source or "persistence" in source:
                label = "Traffic Signal"
                pdf.set_text_color(150, 80, 0)         # amber for traffic signals
            elif "Conditions align" in source or "Conditions remain" in source:
                label = "Prediction Alignment"
                pdf.set_text_color(100, 0, 150)        # purple for prediction
            elif "Peak period" in source or "Off-peak" in source:
                label = "Temporal Factor"
                pdf.set_text_color(80, 80, 80)          # gray for time
            elif "Weather" in source or "rainfall" in source.lower():
                label = "Weather Factor"
                pdf.set_text_color(0, 100, 150)         # teal for weather
            else:
                label = "Policy Rule"
                pdf.set_text_color(100, 100, 100)       # gray for rules

            pdf.set_font("Arial", 'B', 9)
            pdf.cell(35, 6, f"[{label}]", ln=False)
            pdf.set_font("Arial", 'I', 9)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(0, 6, source)

    else:
        pdf.set_font("Arial", size=9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 10, "No specific historical case match. Using default city protocols.", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # 5. MANAGEMENT VERDICT
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Management Verdict", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, decision)

    return pdf.output(dest='S').encode('latin-1')


# ── Unicode cleaner for FPDF ──────────────────────
def clean_for_pdf(text):
    replacements = {
        '\u2248': '~',    '\u2192': '->',   '\u2190': '<-',
        '\u2022': '-',    '\u2014': '--',   '\u2013': '-',
        '\u2019': "'",    '\u2018': "'",    '\u201c': '"',
        '\u201d': '"',    '\u2265': '>=',   '\u2264': '<=',
        '\u00b0': 'deg',  '\u00b2': '2',    '\u00b3': '3',
        '\u03b1': 'alpha','\u03b2': 'beta', '\u2260': '!=',
        '\u00b1': '+/-',  '\u00d7': 'x',    '\u00f7': '/',
        '\u2026': '...',  '\u00a0': ' ',    '\u20ac': 'EUR',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode('latin-1', errors='replace').decode('latin-1')

# Final Result Callout
if st.session_state.decision:
    st.divider()

    is_high = (congestion == "HIGH CONGESTION")
    container = st.error if is_high else st.success

    # Clean unicode before PDF generation
    clean_decision = clean_for_pdf(st.session_state.decision)
    clean_context  = [clean_for_pdf(c) for c in context]

    pdf_bytes = generate_pdf_report(
        junction_name,
        input_data['time'],
        input_data,
        clean_decision,   # ← cleaned
        is_high,
        clean_context     # ← cleaned
    )

    with container(f"### Management Verdict for {junction_name}"):
        st.write("Decision successfully synthesized from SAS and RAG Knowledge Base.")
        st.download_button(
            label="📥 Download PDF Audit Report",
            data=pdf_bytes,
            file_name=f"Report_{junction_name}_{input_data['time'].replace(':', '-')}.pdf",
            mime="application/pdf",
            help="Download the formal management record for this scenario."
        )
# def generate_pdf_report(junction, time, input_data, decision, is_high, context_sources):
#     pdf = FPDF()
#     pdf.add_page()
    
#     # 1. HEADER (Color coded)
#     if is_high:
#         pdf.set_fill_color(255, 200, 200) # Light Red
#         title = f"ALARM: HIGH CONGESTION - {junction}"
#     else:
#         pdf.set_fill_color(200, 255, 200) # Light Green
#         title = f"STATUS: NORMAL - {junction}"
#     pdf.set_font("Arial", 'B', 14)
#     pdf.cell(0, 15, title, ln=True, align='C', fill=True)
#     pdf.ln(5)

#     # 2. METADATA
#     pdf.set_font("Arial", size=10)
#     pdf.cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
#     pdf.cell(0, 8, f"Scenario Time: {time}", ln=True)
#     pdf.ln(5)

#     # 3. SAS DATA SUMMARY
#     pdf.set_font("Arial", 'B', 12)
#     pdf.cell(0, 10, "SAS Predictive Metrics", ln=True)
#     pdf.set_font("Arial", size=11)
#     # Ensure this block only contains the numbers, NOT the decision
#     stats = (f"- State: {input_data['congestion']}\n"
#              f"- Moving Avg: {input_data['moving_avg']:.2f}\n"
#              f"- Lag Traffic: {input_data['lag_traffic']:.2f}\n"
#              f"- Rainfall: {input_data['rainfall']} mm")
#     pdf.multi_cell(0, 8, stats)
#     pdf.ln(5)
#     # Thresholds references
#     pdf.set_font("Arial", 'B', 11)
#     pdf.cell(0, 10, "Statistical Benchmarks (Thresholds)", ln=True)
#     pdf.set_font("Arial", size=10)
#     pdf.multi_cell(0, 7, f"- P50 (Median): {p50_moving:.2f}\n- P75 (Critical): {p75_moving:.2f}")
#     pdf.ln(5)

#     # 4. ADD YOUR NEW SECTION HERE (Historical Reasoning Sources)
#     pdf.set_font("Arial", 'B', 12)
#     pdf.set_text_color(100, 100, 100) # Dark grey for "Sources"
#     pdf.cell(0, 10, "Historical Reasoning Sources (CBR Layer)", ln=True)
#     pdf.set_font("Arial", 'I', 9)

#     if context_sources:
#         for source in context_sources:
#             # This cleans up the JSON string for the PDF
#             pdf.multi_cell(0, 6, f"Ref Case Match: {source}")
#     else:
#         pdf.cell(0, 10, "No specific historical case match. Using default city protocols.", ln=True)
    
#     pdf.set_text_color(0, 0, 0) # Reset to black
#     pdf.ln(5)

#     # 5. MANAGEMENT VERDICT (The Decision)
#     # Ensure pdf.multi_cell(0, 6, decision) only appears ONCE here
#     pdf.set_font("Arial", 'B', 12)
#     pdf.cell(0, 10, "Management Verdict", ln=True)
#     pdf.set_font("Arial", size=10)
#     pdf.multi_cell(0, 6, decision) # <--- The only place the decision is printed

#     return pdf.output(dest='S').encode('latin-1')
   

# # Final Result Callout (Upgraded for Research Audit)
# if st.session_state.decision:
#     st.divider()
    
#     # Determine Status for UI
#     is_high = (congestion == "HIGH CONGESTION")
#     container = st.error if is_high else st.success

#     # Generate the PDF bytes
#     pdf_bytes = generate_pdf_report(
#         junction_name, 
#         input_data['time'], 
#         input_data, 
#         st.session_state.decision, 
#         is_high,
#         context
#     )

#     # Display the Result Box
#     with container(f"### Management Verdict for {junction_name}"):
#         st.write("Decision successfully synthesized from SAS and RAG Knowledge Base.")
        
#         # The Professional PDF Download Button
#         st.download_button(
#             label="📥 Download PDF Audit Report",
#             data=pdf_bytes,
#             file_name=f"Report_{junction_name}_{input_data['time'].replace(':', '-')}.pdf",
#             mime="application/pdf",
#             help="Download the formal management record for this scenario."
#         )
# ===============================
# 📊 CONGESTION HEATMAP
# ===============================
with st.expander("🌡️ City-Wide Congestion Pattern — All Junctions & Hours"):
    

    # Step 1 — compute mean correctly
    heat = df.groupby(["Junction", "Hour"])["I_Congestion"].mean().reset_index()
    
    # Step 2 — debug check (remove after fix confirmed)
    # st.write("Sample heat values:", heat.head(10))
    # st.write("Max value:", heat["I_Congestion"].max())

    # Step 3 — pivot
    heat_pivot = heat.pivot(
        index="Junction", 
        columns="Hour", 
        values="I_Congestion"
    )

    # Step 4 — rename index
    heat_pivot.index = [
        {1:"J1 — Highway", 2:"J2 — City Center",
         3:"J3 — Commercial", 4:"J4 — Residential"}.get(i, f"J{i}")
        for i in heat_pivot.index
    ]

    fig = px.imshow(
        heat_pivot,
        color_continuous_scale="RdYlGn_r",
        zmin=0, zmax=1,
        title="Average Congestion Rate by Hour and Junction",
        labels=dict(x="Hour of Day", y="Junction", color="Rate"),
        text_auto=".2f",
        aspect="auto"
    )
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, width="stretch")

    peak_hour = df.groupby("Hour")["I_Congestion"].mean().idxmax()
    worst_junction = df.groupby("Junction")["I_Congestion"].mean().idxmax()
    st.caption(
        f"📍 Peak congestion hour: **{peak_hour}:00**  "
        f"·  Highest congestion junction: **Junction {worst_junction}**  "
        f"·  Based on {len(df):,} scenarios"
    )

# ===============================
# 📊 DATA PREVIEW
# ===============================
with st.expander("📊 Data Preview"):
    st.dataframe(df.head())

# df = pd.read_csv("scored_data.csv")
# print(df["I_Congestion"].value_counts())
# print(df["I_Congestion"].dtype)
# print(df["Congestion"].value_counts())