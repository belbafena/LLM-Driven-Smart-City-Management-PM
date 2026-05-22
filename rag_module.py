import json
import pandas as pd

# ===============================
# 📊 Load dataset (for thresholds)
# ===============================
df = pd.read_csv("scored_data.csv")

p50_moving = df["MovingAvg3hr"].quantile(0.5)
p75_moving = df["MovingAvg3hr"].quantile(0.75)

p50_lag = df["LagTraffic"].quantile(0.5)
p75_lag = df["LagTraffic"].quantile(0.75)


# ===============================
# 📚 Load knowledge base
# ===============================
def load_knowledge():
    with open("knowledge_base/rules.json") as f:
        rules = json.load(f)
    with open("knowledge_base/cases.json") as f:
        cases = json.load(f)
    with open("knowledge_base/strategies.json") as f:
        strategies = json.load(f)
    return rules, cases, strategies


# ===============================
# 🔎 Threshold categorization
# ===============================
def categorize(value, p50, p75):
    if value > p75:
        return "above_p75"
    elif value > p50:
        return "between_p50_p75"
    else:
        return "below_p50"


# ===============================
# 🔎 Retrieve context
# ===============================
def retrieve_context(input_data):
    rules, cases, strategies = load_knowledge()

    primary_context = []
    secondary_context = []

    moving_avg = input_data["moving_avg"]
    lag = input_data["lag_traffic"]
    rain = input_data["rainfall"]
    hour = int(input_data["time"][:2])
    congestion = input_data["congestion"]

    # ===============================
    # 🔴 PRIMARY DRIVERS (ML-aligned)
    # ===============================

    moving_cat = categorize(moving_avg, p50_moving, p75_moving)
    lag_cat = categorize(lag, p50_lag, p75_lag)

    # MovingAvg → Traffic Intensity
    if moving_cat == "above_p75":
        primary_context.append(
            "High traffic intensity (above P75 threshold), indicating strong demand"
        )
    elif moving_cat == "between_p50_p75":
        primary_context.append(
            "Moderate traffic intensity (between P50 and P75 thresholds)"
        )
    else:
        primary_context.append(
            "Low traffic intensity (below P50 threshold), indicating stable conditions"
        )

    # LagTraffic → Persistence (NOT cause)
    if lag_cat == "above_p75":
        primary_context.append(
            "High traffic persistence (above P75 threshold), indicating sustained traffic patterns"
        )
    elif lag_cat == "between_p50_p75":
        primary_context.append(
            "Moderate traffic persistence, suggesting partial influence from previous conditions"
        )
    else:
        primary_context.append(
            "Low traffic persistence, indicating minimal carry-over from previous conditions"
        )

    # ===============================
    # 🔴 RULE-BASED CONTEXT
    # ===============================
    for rule in rules:
        if rule["importance"] == "primary":
            primary_context.append(f"{rule['description']}")

    # ===============================
    # 🔴 PREDICTION ALIGNMENT
    # ===============================
    if congestion == "HIGH CONGESTION":
        primary_context.append("Conditions align with high congestion patterns")
    else:
        primary_context.append("Conditions remain within normal traffic range")

    # ===============================
    # 🟡 SECONDARY CONTEXT
    # ===============================

    # Time effect
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        secondary_context.append("Peak period increases traffic demand")
    else:
        secondary_context.append("Off-peak period with lower traffic demand")

    # Weather effect
    if rain > 5:
        secondary_context.append("Heavy rainfall significantly impacts traffic flow")
    elif rain > 2:
        secondary_context.append("Moderate rainfall slightly affects traffic conditions")
    else:
        secondary_context.append("Weather conditions have minimal impact")

    # ===============================
    # 🔵 CASE-BASED REASONING
    # ===============================
    for case in cases:
        cond = case["conditions"]

        if (
            cond.get("moving_avg") == moving_cat and
            cond.get("lag_traffic") == lag_cat
        ):
            secondary_context.append(
                f"Case ID: {case['case_id']} - {case['scenario']}"
            )
            secondary_context.append(
                f"Historical action: {case['action_taken']}"
            )
            break

    # ===============================
    # 🟣 STRATEGY RECOMMENDATION
    # ===============================
    rainfall_cat = "present" if rain > 2 else "none"
    for strategy in strategies:
        trig = strategy["trigger"]

        if (
            trig.get("moving_avg") == moving_cat and
            (trig.get("lag_traffic") == lag_cat or "lag_traffic" not in trig) and
            (trig.get("rainfall") == rainfall_cat or "rainfall" not in trig)
        ):
            secondary_context.append(
                f"Recommended strategy: {', '.join(strategy['actions'])}"
            )
            break

    # ===============================
    # ✅ FINAL CONTEXT (PRIORITIZED)
    # ===============================
    return primary_context + secondary_context[:3]