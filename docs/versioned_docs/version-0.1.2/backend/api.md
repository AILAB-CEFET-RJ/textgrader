---
sidebar_position: 2
---

# API

Let's check Textgrader **API** in 5 minutes.

## API Endpoints

Textgrader API is a simple [FastAPI](https://fastapi.tiangolo.com/) project with 3 endpoints:

- Backend API base endpoint: [http://localhost:8000/](http://localhost:8000/)
- Backend API endpoint to grade essays: [http://localhost:8000/text_grade](http://localhost:8000/text_grade)
- Backend API endpoint to API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Base Endpoint

This endpoint just return a message reporting that the API is ok, the code is highlighted bellow:

```py title="backend/src/api.py" {4-10}
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
```

### Text Grade Endpoint

This is the API main endpoint, this endpoints call ``predict_from_text`` function handles giving grades to essays on real time. This endpoint has a ``Request``class that makes the API call validation and reports for the client what kind of data and format he needs to use. This endpoint is highlighted in the code bellow:

```py title="backend/src/api.py"
# highlight-start
class Request(BaseModel):
  essay: str
# highlight-end

@app.get("/")
def home():
  response = {
    "message": "Text Grade API OK! For help go to /docs endpoint."
  }

  return response

# highlight-start
@app.post("/text_grade/")
async def text_grade(request: Request) -> dict[str, int]:
  response = {
    "grade": predict_from_text(request.essay)
  }

  return response
# highlight-end
```
