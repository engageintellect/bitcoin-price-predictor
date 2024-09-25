from fastapi import FastAPI, HTTPException
import json
import os
import uvicorn  # Import Uvicorn

app = FastAPI()

json_file = './data/btc_price_predictions.json'

@app.get("/predictions")
async def get_predictions():
    if not os.path.exists(json_file):
        raise HTTPException(status_code=404, detail="Prediction file not found.")
    
    with open(json_file, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Error reading prediction data.")
    
    return data

# Run Uvicorn when this script is executed directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=4325, reload=True)

