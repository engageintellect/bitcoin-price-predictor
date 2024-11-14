from fastapi import FastAPI, HTTPException
import json
import os
import uvicorn  
import requests

app = FastAPI()

json_file = '../data/btc_price_predictions.json'

@app.get("/predictions")
async def get_predictions():
    if not os.path.exists(json_file):
        raise HTTPException(status_code=404, detail="Prediction file not found.")

    with open(json_file, 'r') as file:
        try:
            data = json.load(file)

            # Analyze the correctness of the predictions
            correct_predictions = 0
            total_predictions = len(data) - 1  # We exclude the last item as there's no "next day" to compare

            for i in range(total_predictions):
                current_day = data[i]
                next_day = data[i + 1]

                # Compare the prediction with the actual change in price
                if current_day["prediction"] == "UP" and next_day["closePrice"] > current_day["closePrice"]:
                    correct_predictions += 1
                elif current_day["prediction"] == "DOWN" and next_day["closePrice"] < current_day["closePrice"]:
                    correct_predictions += 1

            # Calculate the percentage of correct predictions
            accuracy_percentage = (correct_predictions / total_predictions) * 100
            out = f"{accuracy_percentage:.2f}%"

        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Error reading prediction data.")

    # Make sure predictions come first in the output
    return {
        'predictionAccuracy': out,
        'predictions': data,
    }


# Run Uvicorn when this script is executed directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=4325, reload=True)

