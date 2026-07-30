from fastapi import FastAPI

from rag_service.routers import analyze_router, health_router

app = FastAPI()

app.include_router(analyze_router)

app.include_router(health_router)
