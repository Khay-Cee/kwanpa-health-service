# FastAPI Calorie Tracker for Render

This folder contains the minimal files needed to deploy your FastAPI calorie tracker model to Render.

uvicorn app:app --reload

curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@apple.webp"

  
## Files
- `app.py`: FastAPI app with model loading and prediction endpoint
- `requirements.txt`: Python dependencies
- `Procfile`: Render process definition
- `model_trained_101class.keras`: Your trained model file (upload manually)

## Deployment
1. Upload all files to this folder, including your model file.
2. Push to your GitHub repo and connect to Render.
3. Set the start command to use the Procfile or `uvicorn app:app --host=0.0.0.0 --port=10000`.
