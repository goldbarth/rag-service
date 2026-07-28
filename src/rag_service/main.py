from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class TextRequest(BaseModel):
    text: str = Field(min_length=3)
    
class TextResponse(BaseModel):
    result: str
    num_chars: int
    

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"version": "0.1.0"}

@app.post("/analyze", response_model=TextResponse)
def analyze(request: TextRequest):
    return TextResponse(result=request.text, num_chars=len(request.text))
