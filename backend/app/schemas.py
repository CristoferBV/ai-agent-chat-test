from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(min_length=3, description="Pregunta del usuario")

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
