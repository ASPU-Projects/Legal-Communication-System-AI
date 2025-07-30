from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from main import recommendation

app = FastAPI(title='Legal Communication System',version='1.0.1')


# Add CORS Middleware to handle preflight requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods including OPTIONS
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def Ready():
    return {"message":"ready"}

@app.get("/get_recommanded/{user_text}")
def run_get_recommanded(user_text):
    return recommendation(user_text)

# Explicitly allow OPTIONS request (optional)
@app.options("/get_recommanded/{user_text}")
def options_handler(user_text):
    return JSONResponse(content=recommendation(user_text), status_code=200)

# uvicorn api:app --reload --host 0.0.0.0 --port 8000 --timeout-keep-alive 120

# http://mohammadkheryadj-64003.portmap.host:64003/get_recommanded/uers_text