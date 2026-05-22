from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
# if api_key:
#     print("✅ API Key loaded successfully!")
# else:
#     print("❌ API Key NOT found. Check your file name and variable name.")

def generate_decision(input_data, context, confidence):

    prompt = f"""
You are an intelligent decision-support system for smart city traffic management.

Your role is to analyze traffic conditions and provide clear, data-driven, and actionable decisions.

==============================
INPUT DATA
==============================
- Congestion Level: {input_data['congestion']}
- Moving Average Traffic (3hr): {input_data['moving_avg']}
- Lag Traffic: {input_data['lag_traffic']}
- Rainfall: {input_data['rainfall']} mm
- Time: {input_data['time']}

==============================
DECISION PRIORITY LOGIC
==============================
1. Traffic persistence (moving average, lag traffic) = PRIMARY driver
2. Time (peak hours) = SECONDARY factor
3. Weather (rainfall) = MINOR factor

- Interpret moving average as traffic intensity.
- Interpret lag traffic as traffic persistence (NOT a direct cause).
- Use terms: "low persistence", "moderate persistence", "high persistence".
- Prioritize persistence-based reasoning in explanations.

==============================
THRESHOLD REFERENCE
==============================
- MovingAvg P50: 15.33
- MovingAvg P75: 29.33
- LagTraffic P50: 15.00
- LagTraffic P75: 29.00

==============================
THRESHOLD INTERPRETATION RULE
==============================
- Compare all values to P50 and P75 thresholds.
- below P50 → low traffic
- between P50–P75 → moderate traffic
- above P75 → high congestion risk
- Clearly link threshold level to required action (below P50 → no intervention, above P75 → active control)
- If all indicators are below P50, explicitly state that no congestion mitigation is required.

==============================
CONTEXT (RAG OUTPUT)
==============================
{chr(10).join(context)}

==============================
MODEL RELIABILITY (SAS)
==============================
- Confidence Score: {confidence:.2f}
- Reliability Level: {"HIGH" if confidence > 0.8 else "MODERATE" if confidence > 0.6 else "LOW"}
- Prediction: {"HIGH CONGESTION" if input_data['congestion'] == "HIGH CONGESTION" else "NORMAL TRAFFIC"}
- Confidence Meaning: {"This score reflects confidence that congestion IS occurring" if input_data['congestion'] == "HIGH CONGESTION" else "This score reflects confidence that traffic IS normal"}

==============================
CONFIDENCE INSTRUCTION
==============================
- Reflect reliability level in explanation.

HIGH:
- Use confident but realistic language ("strongly suggests", "highly reliable")
- Even with high confidence, avoid definitive conclusions
- Do NOT imply certainty; use "highly reliable indication" instead
- Avoid absolute claims ("certain", "guaranteed")

MODERATE:
- Use cautious language ("likely", "suggests")
- Recommend balanced actions

LOW:
- Use uncertainty language ("uncertain", "borderline")
- Focus on monitoring and validation
- Avoid aggressive actions

==============================
OUTPUT RULES
==============================
- Do NOT assume measurement units
- Use "traffic intensity" instead of physical units
- Avoid vague or unsupported statements
- Use precise, technical language
- Avoid repetition
- Use proper Markdown formatting
- Do NOT place text on same line as headings
- Recommended actions must be directly related to traffic control operations (e.g., signal timing, monitoring, traffic updates)
- Avoid generic or planning-oriented actions (e.g., "data collection", "prepare for future")

==============================
CONDITIONAL BEHAVIOR
==============================
NORMAL TRAFFIC:
- Maintain operations and monitoring
- Avoid unnecessary interventions

HIGH CONGESTION:
- Recommend active traffic control measures

==============================
TASK
==============================

### 1. Traffic Situation
Explain current conditions using input data and thresholds.
Prioritize traffic persistence.
Include confidence and describe reliability level.
Do NOT claim absolute certainty.

### 2. Recommended Actions
Provide 2–3 realistic traffic management actions.

### 3. Justification
Justify actions using:
- Threshold comparisons (P50/P75)
- Model confidence (include value, e.g., ≈0.75)
- RAG context (mention Case ID if available)
- Justification must explicitly connect threshold values → traffic condition → recommended action

==============================
STYLE
==============================
- Professional and concise
- Clear, structured explanation
- Use technical terminology (traffic intensity, persistence)
- Avoid overgeneralization
- Avoid repeating strong or identical wording (e.g., "necessitating"); vary phrasing such as "indicating the need for"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content