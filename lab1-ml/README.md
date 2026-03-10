# Lab 1: Dockerized Sentiment Analysis API

## Overview

A pretrained sentiment analysis model served via a REST API using Flask and Waitress, containerized with Docker.

- **Model:** distilbert-base-uncased-finetuned-sst-2-english (HuggingFace)
- **Server:** Waitress
- **Port:** 8080

## Requirements

- Docker installed on your machine

## How to Run

### Option 1: Pull from Docker Hub (Recommended)

```bash
docker pull stupnd/ml-sentiment-api:latest
docker run -p 8080:8080 stupnd/ml-sentiment-api:latest
```

### Option 2: Build Locally

```bash
docker build -t ml-sentiment-api:latest .
docker run -p 8080:8080 ml-sentiment-api:latest
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8080/
```

### Predict Sentiment

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'
```

**Example Response:**

```json
{
  "input_text": "This is amazing!",
  "prediction": {
    "label": "POSITIVE",
    "score": 0.9999
  }
}
```

## Docker Hub

[https://hub.docker.com/r/stupnd/ml-sentiment-api](https://hub.docker.com/r/stupnd/ml-sentiment-api)

## Files

- **app.py** — Flask application and model inference code
- **requirements.txt** — Python dependencies
- **Dockerfile** — Container configuration
