import os
import json
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from models import EmergencyInput, EmergencyPlan

# ✅ Correct initialization - model name as first positional argument
model = OpenAIChatModel(
    "mistralai/mixtral-8x22b-instruct",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

agent = Agent(
    model=model,
    system_prompt="""
You are an emergency response assistant.

You must ONLY handle:
- fire
- flood
- earthquake

Respond ONLY in valid JSON:
{
  "immediate_actions": [],
  "do_not_do": [],
  "evacuation_decision": "",
  "escalation_guidance": "",
  "safety_disclaimer": ""
}
"""
)

async def generate_emergency_plan(data: EmergencyInput) -> EmergencyPlan:
    prompt = f"""
Emergency type: {data.emergency_type}
Immediate danger: {data.immediate_danger}
"""

    try:
        result = await agent.run(prompt)
        parsed = json.loads(result.output)
        return EmergencyPlan(**parsed)

    except Exception:
        # ✅ HARD FAILSAFE with ALL required fields
        return EmergencyPlan(
            immediate_actions=["Move to safety", "Follow authorities"],
            do_not_do=["Do not panic"],
            evacuation_decision="Follow local guidance",
            escalation_guidance="Contact emergency services if in immediate danger",  # ✅ ADDED
            safety_disclaimer="General guidance only, not emergency services"
        )