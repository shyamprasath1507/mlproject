#!/bin/bash
set -e

echo "Starting AI Engineering Change Impact Analyzer Setup..."

# 1. Generate Mock Data
echo "Generating mock ECO data..."
python backend/generate_data.py

# 2. Train Models
echo "Training ML models..."
python backend/train_model.py

# 3. Start FastAPI in the background
echo "Starting FastAPI server on port 8000..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Wait a moment to let the backend start up
sleep 3

# 4. Start Streamlit in the foreground
echo "Starting Streamlit frontend on port 8501..."
exec streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0
