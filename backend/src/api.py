from fastapi import FastAPI

from pydantic import BaseModel

from fastapi.responses import ORJSONResponse

from fastapi.middleware.cors import CORSMiddleware

from dags.predict.predict_from_text import predict_from_text

from save_essay import save

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
    "message": "OK!"
  }

  return response

@app.post("/grade")
async def text_grade(request: Request) -> dict[str, float]:
  grade = predict_from_text(request.essay)
  #Database().save_data(request.essay, grade)
  save({"essay": request.essay, "grade": grade})

  return {"grade": grade}
