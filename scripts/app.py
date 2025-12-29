import os
import sys

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from inference import generate_soap_note

    MODEL_AVAILABLE = True
except Exception as e:
    print(
        f"Warning: Could not import inference module ({e}). API will use mock responses."
    )
    MODEL_AVAILABLE = False

    def generate_soap_note(dialogue, use_model=False):
        return (
            "Subjective: Patient reports symptoms.\n"
            "Objective: Physical exam findings.\n"
            "Assessment: Clinical diagnosis.\n"
            "Plan: Treatment recommendations."
        )


app = FastAPI(
    title="MediSum API",
    description="API for generating clinical SOAP notes from dialogues using fine-tuned Llama-3-8B",
)


class DialogueRequest(BaseModel):
    dialogue: str


class SOAPResponse(BaseModel):
    soap_note: str
    model_used: str


@app.post("/generate", response_model=SOAPResponse)
async def generate(request: DialogueRequest):
    """Generate SOAP note from doctor-patient dialogue"""
    try:
        if not request.dialogue.strip():
            raise HTTPException(status_code=400, detail="Dialogue cannot be empty")

        # Try to use real model, fallback to mock if not available
        soap_note = generate_soap_note(request.dialogue, use_model=MODEL_AVAILABLE)
        model_used = "fine-tuned-llama3-8b" if MODEL_AVAILABLE else "mock"

        return SOAPResponse(soap_note=soap_note, model_used=model_used)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
