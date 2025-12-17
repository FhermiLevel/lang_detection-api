# Language Detection API

A FastAPI-based machine learning API that detects the language of input text with confidence scores.  
Supports both single-text and batch predictions.

---

## Features
- Language detection using a trained ML pipeline
- Confidence score for each prediction
- Batch prediction support
- Interactive API documentation (Swagger UI)

---

## Tech Stack
- Python
- FastAPI
- Scikit-learn
- Joblib
- Uvicorn

---

## Setup Instructions

### 1. Create and activate virtual environment
```bash
conda create -n fastapi_env python=3.10
conda activate fastapi_env
```
### 2. Install dependencies
```
pip install fastapi uvicorn scikit-learn joblib numpy
```
### 3. Run the API
```
uvicorn main:app --reload
```
### API Endpoints
Root
GET /
Returns a welcome message.

#### Single Text Prediction
POST /predict

###### Request Body
```
{
  "text": "Bonjour tout le monde"
}
```
###### Response
```
{
  "predicted_language": "French",
  "confidence": 0.9999
}
```
#### Batch Prediction
POST /predict-batch

###### Request Body
```
{
  "texts": [
    "Hello world",
    "Bonjour le monde",
    "Hola mundo"
  ]
}
```
###### Response
```
{
  "results": [
    {
      "text": "Hello world",
      "predicted_language": "English",
      "confidence": 0.7918
    },
    {
      "text": "Bonjour le monde",
      "predicted_language": "French",
      "confidence": 0.9963
    },
    {
      "text": "Hola mundo",
      "predicted_language": "Spanish",
      "confidence": 0.6976
    }
  ]
}

```
#### Use Cases

- NLP preprocessing pipelines
- Language-based routing systems
- Educational tools
- Multilingual content moderation

#### Author
***Developed by Oluwadare Olalekan***

