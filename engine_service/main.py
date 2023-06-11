from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from engine_service.api.auth import auth_router


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(auth_router)