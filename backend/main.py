from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import EmergencyInput, EmergencyPlan
from agent import generate_emergency_plan

app = FastAPI(title="AI Emergency Decision Agent")

# ✅ CORS FIX (THIS IS THE IMPORTANT PART)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/plan", response_model=EmergencyPlan)
async def create_plan(input_data: EmergencyInput):
    return await generate_emergency_plan(input_data)
@app.post("/escalate")
async def escalate(payload: dict):
    return {
        "decision": "Escalation triggered",
        "received": payload
    }

