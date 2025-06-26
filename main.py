from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

app = FastAPI()

HF_API_KEY = os.getenv("HF_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Chat Assistant
class ChatRequest(BaseModel):
    question: str

@app.post("/chat/ask")
def ask_chat(req: ChatRequest):
    resp = requests.post(
        "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
        headers={
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "inputs": f"[INST] {req.question} [/INST]",
            "parameters": {"max_new_tokens": 250, "temperature": 0.7}
        }
    )
    return {"answer": resp.json()[0].get("generated_text")} if resp.status_code == 200 else {"error": resp.text}

# Policy summarizer
class PolicyRequest(BaseModel):
    text: str

@app.post("/policy/summarize")
def summarize_policy(req: PolicyRequest):
    resp = requests.post(
        "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": req.text}
    )
    return {"summary": resp.json()[0].get("summary_text")} if resp.status_code == 200 else {"error": resp.text}

# Eco tips
@app.get("/eco/tips")
def get_eco_tips():
    return {"tips": [
        "Turn off lights when not in use.",
        "Use public transport or carpool.",
        "Install low-flow shower heads.",
        "Separate recyclables and wet waste.",
        "Switch to LED lighting."
    ]}

# Weather service
@app.get("/weather/get")
def get_weather(city: str):
    resp = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    )
    if resp.status_code == 200:
        data = resp.json()
        return {
            "Temperature": f"{data['main']['temp']}°C",
            "Humidity": f"{data['main']['humidity']}%",
            "Weather": data['weather'][0]['description'].title()
        }
    else:
        return {"error": f"Weather data not available for '{city}'"}
