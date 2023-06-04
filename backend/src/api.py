from fastapi import FastAPI

from pydantic import BaseModel

from fastapi.responses import ORJSONResponse

from fastapi.middleware.cors import CORSMiddleware

from dags.predict.predict_from_text import predict_from_text

app = FastAPI(default_response_class=ORJSONResponse)

origins = [
  "*",
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

class Request(BaseModel):
  essay: str

@app.get("/")
def home():
  response = {
    "message": "Text Grade API OK! For help go to /docs endpoint."
  }

  return response

@app.post("/text_grade/")
async def text_grade(request: Request) -> dict[str, int]:
  response = {
    "grade": predict_from_text(request.essay)
  }

  return response
