FROM python:3.10-slim

# Install required packages
RUN pip install --no-cache-dir pyflowmeter pandas numpy scikit-learn joblib

# Set working directory
WORKDIR /app

# Copy the saved pickle files and deployment script
COPY . .

# Default command (can be overridden)
CMD ["python", "deploy.py"]
