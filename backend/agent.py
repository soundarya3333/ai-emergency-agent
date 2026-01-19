import os
import json
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from models import EmergencyInput, EmergencyPlan

# ❌ DO NOT rely on load_dotenv on Render
# load_dotenv() is fine locally but irrelevant in production

model = OpenAIChatModel(
    model="mistralai/mixtral-8x22b-instruct",
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
        # HARD FAILSAFE (Render-safe)
        return EmergencyPlan(
            immediate_actions=["Move to safety", "Follow authorities"],
            do_not_do=["Do not panic"],
            evacuation_decision="Follow local guidance",
            safety_disclaimer="General guidance only, not emergency services"
        )
