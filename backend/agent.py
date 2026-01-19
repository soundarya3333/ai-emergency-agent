import os
import json
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from models import EmergencyInput, EmergencyPlan

# ─── IMPORTANT: Tell pydantic-ai to use OpenRouter ──────────────────────────
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# Now create model — only model name, no extra kwargs!
model = OpenAIChatModel("mistralai/mixtral-8x22b-instruct")

agent = Agent(
    model=model,
    system_prompt="""
You are an emergency response assistant.

You must ONLY handle:
- fire
- flood
- earthquake

You MUST respond ONLY in valid JSON with the following structure:

{
  "immediate_actions": ["..."],
  "do_not_do": ["..."],
  "evacuation_decision": "...",
  "escalation_guidance": "...",
  "safety_disclaimer": "..."
}

Rules:
- Give clear, calm, practical instructions
- Do NOT give medical advice
- Do NOT predict outcomes
- Always prioritize personal safety
"""
)

async def generate_emergency_plan(data: EmergencyInput) -> EmergencyPlan:
    prompt = f"""
Emergency type: {data.emergency_type}
Immediate danger present: {data.immediate_danger}

Generate the emergency response JSON.
"""

    try:
        result = await agent.run(prompt)
        parsed = json.loads(result.output)
        return EmergencyPlan(**parsed)
    except Exception as e:
        print("AI generation failed:", str(e))  # ← helps debugging on Render
        return EmergencyPlan(
            immediate_actions=[
                "Move to a safe location",
                "Follow official instructions"
            ],
            do_not_do=[
                "Do not panic",
                "Do not spread unverified information"
            ],
            evacuation_decision="Follow local authority guidance",
            escalation_guidance="Contact emergency services if in immediate danger",
            safety_disclaimer="This is general guidance only and not a substitute for emergency services."
        )