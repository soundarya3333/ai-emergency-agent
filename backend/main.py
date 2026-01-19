from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from models import EmergencyInput, EmergencyPlan
from agent import generate_emergency_plan
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Emergency Decision Agent",
    description="AI-powered emergency response guidance system",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint - returns service status"""
    return {
        "status": "online",
        "service": "AI Emergency Decision Agent",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "Service status and API information",
            "GET /health": "Health check endpoint",
            "POST /plan": "Generate emergency response plan",
            "POST /escalate": "Escalate emergency situation"
        },
        "supported_emergencies": ["fire", "flood", "earthquake"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "ai-emergency-agent"
    }

@app.post("/plan", response_model=EmergencyPlan)
async def create_plan(input_data: EmergencyInput):
    """
    Generate an emergency response plan based on the situation.
    
    Args:
        input_data: Emergency type and immediate danger status
        
    Returns:
        EmergencyPlan with actionable guidance
    """
    try:
        logger.info(f"Generating plan for {input_data.emergency_type} (immediate_danger={input_data.immediate_danger})")
        plan = await generate_emergency_plan(input_data)
        logger.info("Plan generated successfully")
        return plan
    except Exception as e:
        logger.error(f"Error generating plan: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate emergency plan: {str(e)}")

@app.post("/escalate")
async def escalate(payload: dict):
    """
    Escalate an emergency situation.
    
    Args:
        payload: Emergency escalation data
        
    Returns:
        Confirmation of escalation
    """
    logger.info(f"Escalation triggered: {payload}")
    return {
        "status": "escalated",
        "decision": "Emergency escalation triggered",
        "received": payload,
        "action": "Emergency services should be contacted immediately via 911"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "message": "Please check the API documentation at /",
        "path": str(request.url)
    }

@app.exception_handler(500)
async def server_error_handler(request, exc):
    logger.error(f"Server error: {exc}")
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again."
    }