FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY backend/ backend/
COPY frontend/ frontend/
COPY entrypoint.sh .

# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Run the entrypoint script
ENTRYPOINT ["./entrypoint.sh"]
