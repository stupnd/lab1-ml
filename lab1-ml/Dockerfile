# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose port 8080
EXPOSE 8080

# Set production environment variable
ENV PRODUCTION=true

# Run the application
CMD ["python", "app.py"]