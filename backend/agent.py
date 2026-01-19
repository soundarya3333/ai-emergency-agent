import os
import json
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from models import EmergencyInput, EmergencyPlan

# Get API key from environment (set in Render)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set in Render")

print(f"✓ API Key loaded: {OPENAI_API_KEY[:15]}...")
print(f"✓ Base URL: {OPENAI_BASE_URL}")

# Configure OpenRouter model
model = OpenAIModel(
    "mistralai/mistral-7b-instruct:free",
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)

agent = Agent(
    model=model,
    result_type=EmergencyPlan,
    system_prompt="""
You are an emergency response assistant.

You must ONLY handle: fire, flood, earthquake

Provide clear, calm, practical instructions in JSON format with these exact fields:
- immediate_actions: array of 3-5 urgent steps to take
- do_not_do: array of 3-5 things to avoid
- evacuation_decision: clear yes/no decision with brief reasoning
- escalation_guidance: when to call emergency services
- safety_disclaimer: brief legal disclaimer

Rules:
- Be concise and actionable
- Do NOT give medical advice
- Do NOT predict outcomes
- Always prioritize personal safety
- Use simple, clear language
"""
)

async def generate_emergency_plan(data: EmergencyInput) -> EmergencyPlan:
    """
    Generate emergency response plan using AI.
    """
    prompt = f"""
Emergency situation:
- Type: {data.emergency_type}
- Immediate danger present: {data.immediate_danger}

Generate a clear, actionable emergency response plan in JSON format.
"""

    try:
        print(f"Generating plan for {data.emergency_type}...")
        result = await agent.run(prompt)
        print("✓ Plan generated successfully")
        return result.data
    
    except Exception as e:
        print(f"⚠ AI generation failed: {e}")
        print("Using fallback response...")
        
        # Safe fallback response based on emergency type
        fallback_actions = {
            "fire": [
                "Exit the building immediately using the nearest safe exit",
                "Stay low to avoid smoke inhalation",
                "Feel doors before opening - if hot, use another route",
                "Once outside, move away from the building",
                "Call 911 from a safe location"
            ],
            "flood": [
                "Move to higher ground immediately",
                "Avoid walking or driving through flood waters",
                "Turn off electricity at main breaker if safe to do so",
                "Listen to emergency broadcasts for evacuation orders",
                "Do not return home until authorities say it's safe"
            ],
            "earthquake": [
                "DROP to hands and knees to prevent being knocked down",
                "Take COVER under a sturdy desk or table",
                "HOLD ON to your shelter and be ready to move with it",
                "Stay away from windows and objects that could fall",
                "If outdoors, move to open area away from buildings"
            ]
        }
        
        fallback_donts = {
            "fire": [
                "Do not use elevators",
                "Do not go back inside for belongings",
                "Do not open doors that are hot to touch",
                "Do not break windows unless necessary for escape",
                "Do not waste time gathering items"
            ],
            "flood": [
                "Do not walk through moving water (6 inches can knock you down)",
                "Do not drive through flooded areas (2 feet can sweep away vehicles)",
                "Do not touch electrical equipment if wet or standing in water",
                "Do not drink flood water without purification",
                "Do not ignore evacuation orders"
            ],
            "earthquake": [
                "Do not run outside during shaking",
                "Do not stand in doorways (not safer than other locations)",
                "Do not use elevators",
                "Do not light matches if you smell gas",
                "Do not use your phone except for emergencies"
            ]
        }
        
        evacuation_decisions = {
            "fire": "YES - Evacuate immediately. Do not attempt to fight the fire unless it is very small and you have proper training.",
            "flood": "YES if water is rising or evacuation is ordered. Move to higher ground immediately. If not ordered, prepare to evacuate.",
            "earthquake": "NO during shaking - take cover. YES after shaking stops if building is damaged or you smell gas."
        }
        
        return EmergencyPlan(
            immediate_actions=fallback_actions.get(data.emergency_type, [
                "Move to a safe location immediately",
                "Call emergency services (911)",
                "Follow instructions from authorities",
                "Stay calm and help others if safe to do so"
            ]),
            do_not_do=fallback_donts.get(data.emergency_type, [
                "Do not panic",
                "Do not put yourself in danger",
                "Do not spread unverified information",
                "Do not ignore official warnings"
            ]),
            evacuation_decision=evacuation_decisions.get(
                data.emergency_type,
                "Follow local authority guidance. Evacuate if ordered or if you feel unsafe."
            ),
            escalation_guidance="Call 911 immediately if: you are in immediate danger, someone is injured, you see fire/smoke, or structural damage is present. For non-emergencies, contact local emergency management.",
            safety_disclaimer="This is general guidance only and not a substitute for professional emergency services. Always prioritize your safety and follow official emergency service instructions and local authorities. When in doubt, call 911."
        )