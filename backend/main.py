from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
import random

app = FastAPI(title="AI Engineering Change Impact Analyzer API")

# Global variables to hold models
risk_model = None
cost_model = None
risk_label_encoder = None

class ECORequest(BaseModel):
    complexity_score: float = Field(..., ge=1.0, le=10.0, description="Complexity of the change (1.0 to 10.0)")
    department_count: int = Field(..., ge=1, le=5, description="Number of departments involved (1 to 5)")
    original_cost_usd: float = Field(..., ge=0, description="Original estimated cost of the component/project")
    change_domain: str = Field(..., description="Domain of the change: Mechanical, Electrical, or Software")

@app.on_event("startup")
def load_models():
    global risk_model, cost_model, risk_label_encoder
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    risk_path = os.path.join(base_dir, "risk_model.pkl")
    cost_path = os.path.join(base_dir, "cost_model.pkl")
    encoder_path = os.path.join(base_dir, "risk_label_encoder.pkl")
    
    try:
        risk_model = joblib.load(risk_path)
        cost_model = joblib.load(cost_path)
        risk_label_encoder = joblib.load(encoder_path)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        # Note: In a real production app we might not want to continue if models fail to load, 
        # but for this script we allow startup so the container doesn't instantly crash if models 
        # aren't generated yet (though entrypoint.sh should generate them first).

def generate_mock_downstream_components(domain: str):
    components = {
        "Mechanical": ["Chassis Assembly", "Cooling System", "Mounting Brackets", "Housing", "Bearings"],
        "Electrical": ["Wiring Harness", "Power Supply", "Sensor Array", "PCB Board", "Connectors"],
        "Software": ["Control Firmware", "User Interface Module", "API Gateway", "Database Schema", "Auth Service"]
    }
    
    domain_components = components.get(domain, ["Generic Component A", "Generic Component B"])
    
    # Pick a random subset of 1 to 3 affected components
    num_affected = random.randint(1, 3)
    return random.sample(domain_components, num_affected)

@app.post("/predict")
def predict_eco_impact(eco: ECORequest):
    if risk_model is None or cost_model is None or risk_label_encoder is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Ensure training script has been run.")
        
    # Convert input to DataFrame
    input_data = pd.DataFrame([{
        "complexity_score": eco.complexity_score,
        "department_count": eco.department_count,
        "original_cost_usd": eco.original_cost_usd,
        "change_domain": eco.change_domain
    }])
    
    try:
        # Predict Risk
        risk_pred_encoded = risk_model.predict(input_data)
        predicted_risk = risk_label_encoder.inverse_transform(risk_pred_encoded)[0]
        
        # Predict Cost Overrun
        predicted_cost_overrun = float(cost_model.predict(input_data)[0])
        
        # Mock Downstream Components
        affected_components = generate_mock_downstream_components(eco.change_domain)
        
        return {
            "predicted_risk": predicted_risk,
            "predicted_cost_overrun": max(0, predicted_cost_overrun), # Ensure no negative overrun
            "downstream_affected_components": affected_components
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
