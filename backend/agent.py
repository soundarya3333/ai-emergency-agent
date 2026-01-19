import os
import json
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from models import EmergencyInput, EmergencyPlan

# ─── IMPORTANT: Tell pydantic-ai to use OpenRouter ──────────────────────────
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# Model initialization - only model name
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
- Be concise and to the point
"""
)

async def generate_emergency_plan(data: EmergencyInput) -> EmergencyPlan:
    """
    Generate emergency response plan using AI with maximum 500 tokens limit.
    """
    prompt = f"""
Emergency type: {data.emergency_type}
Immediate danger present: {data.immediate_danger}

Generate the emergency response JSON.
Be extremely concise.
"""

    try:
        result = await agent.run(
            prompt,
            max_tokens=500,           # ← This limits the response length
            temperature=0.3,          # Lower = more focused/consistent
        )
        
        # Clean up any possible extra whitespace
        output_text = result.output.strip()
        
        parsed = json.loads(output_text)
        return EmergencyPlan(**parsed)
    
    except Exception as e:
        print("AI generation failed:", str(e))
        # Fallback response - safe default plan
        return EmergencyPlan(
            immediate_actions=[
                "Move to a safe location immediately",
                "Follow instructions from emergency services"
            ],
            do_not_do=[
                "Do not panic",
                "Do not spread unverified information"
            ],
            evacuation_decision="Follow local authority guidance",
            escalation_guidance="Contact emergency services if in immediate danger",
            safety_disclaimer="This is general guidance only and not a substitute for professional emergency services."
        )