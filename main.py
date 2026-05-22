import pandas as pd
from rag_module import retrieve_context
from llm_module import generate_decision

# Load SAS output
df = pd.read_csv("scored_data.csv")   

# Take one row
row = df.iloc[0]

# Build input from REAL SAS output
input_data = {
    "congestion": "HIGH CONGESTION" if int(row["I_Congestion"]) == 1 else "NORMAL TRAFFIC",
    "moving_avg": row["MovingAvg3hr"],
    "lag_traffic": row["LagTraffic"],
    "rainfall": row["RAIN_MM"],
    "time": f"{int(row['Hour']):02d}:00"
}

# Step 1: RAG
context = retrieve_context(input_data)
confidence = row["EM_PROBABILITY"]

# Step 2: LLM
decision = generate_decision(input_data, context, confidence)
print("\n=== INPUT DATA ===")
print(input_data)

print("\n=== CONTEXT (RAG) ===")
for c in context:
    print("-", c)
print("\n=== FINAL DECISION ===\n")
print(decision)