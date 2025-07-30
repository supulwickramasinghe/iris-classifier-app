# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
COPY model.py .
COPY app.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Train model (runs model.py to create iris_model.pkl)
RUN python model.py

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
