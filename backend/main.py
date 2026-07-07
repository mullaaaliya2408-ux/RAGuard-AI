"""
RAGuard AI - Backend Entry Point

This is the FastAPI application root. It wires together routers,
middleware, and startup/shutdown events. Business logic lives in
services/ and agents/, not here.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from configs.settings import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("raguard")

app = FastAPI(
    title="RAGuard AI",
    description="An AI system that knows when it does not know.",
    version="0.1.0",
)

# Allow the Next.js frontend (localhost:3000) to call this API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Simple liveness check used by monitoring and the frontend dashboard."""
    logger.info("Health check called")
    return {"status": "ok", "service": "RAGuard AI", "version": "0.1.0"}


@app.get("/")
def root():
    return {"message": "RAGuard AI backend is running. See /docs for API reference."}
from api.routes import upload, query, documents, evaluation

app.include_router(upload.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(evaluation.router)