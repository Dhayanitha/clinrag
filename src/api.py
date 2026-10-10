
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag_pipeline import answer_question

# 1. Create the API application

app = FastAPI(
    title="ClinRAG API",
    description="Evidence-grounded clinical knowledge retrieval",
    version="1.0.0"
)


# 2. Define the request format

class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=500,
        description="The clinical knowledge question"
    )


# 3. Define the health-check endpoint

@app.get("/health")
def health_check():
    return {"status": "ok"}


# 4. Define the question endpoint

@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        return answer_question(request.question.strip())
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="The question could not be processed."
        ) from error
