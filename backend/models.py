from enum import Enum
from pydantic import BaseModel
from typing import List


class EmergencyType(str, Enum):
    fire = "fire"
    flood = "flood"
    earthquake = "earthquake"


class EmergencyInput(BaseModel):
    emergency_type: EmergencyType
    immediate_danger: bool


class EmergencyPlan(BaseModel):
    immediate_actions: List[str]
    do_not_do: List[str]
    evacuation_decision: str
    escalation_guidance: str
    safety_disclaimer: str