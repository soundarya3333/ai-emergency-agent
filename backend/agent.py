from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from models import EmergencyInput, EmergencyPlan
import json

load_dotenv()

model = OpenAIChatModel(
    "mistralai/mixtral-8x22b-instruct"
)

agent = Agent(
    model=model,
    system_prompt="""
You are an emergency response assistant.

You must ONLY handle:
- fire
- flood
- earthquake

You MUST respond ONLY in valid JSON with the structure:
{
  "immediate_actions": ["..."],
  "do_not_do": ["..."],
  "evacuation_decision": "...",
  "escalation_guidance": "...",
  "safety_disclaimer": "..."
}
"""
)

async def generate_emergency_plan(data: EmergencyInput) -> EmergencyPlan:
    prompt = f"""
Emergency type: {data.emergency_type}
Immediate danger present: {data.immediate_danger}
Generate the emergency response JSON.
"""

    result = await agent.run(prompt)
    parsed = json.loads(result.output)
    return EmergencyPlan(**parsed)
