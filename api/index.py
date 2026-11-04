from fastapi import FastAPI

app = FastAPI(
    title="NeuroStorm API",
    description="Intelligent Tool Wear Monitoring and Predictive Maintenance",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "project": "NeuroStorm",
        "status": "running",
        "message": "NeuroStorm API is deployed successfully!"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
