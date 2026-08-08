"""
api/main.py — FastAPI application entry point.

Creates the FastAPI app, configures CORS (so the Lovable frontend
can call it from a browser), mounts the route router, and adds a
root health-check endpoint.

Run locally:
    uvicorn api.main:app --reload --port 8000

Interactive API docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.discovery_routes import router as discovery_router
from api.investigation_routes import router as investigation_router

# ── App Instance ──────────────────────────────────────────────────────

app = FastAPI(
    title="AI Cyber Discovery Engine API",
    description=(
        "REST interface for the AI Cyber Discovery Engine. "
        "Accepts any analyst input (IOC, CVE, malware name, APT group, "
        "MITRE technique, natural language, STIX bundle, JSON logs, reports) "
        "and returns normalized, correlated, and explained threat intelligence."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────
# Permissive for local development and Lovable preview.
# Tighten origins list for production deployment.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────

app.include_router(router, prefix="/api/v1", tags=["engine"])
app.include_router(discovery_router, prefix="/api/v1", tags=["discovery"])
app.include_router(investigation_router, prefix="/api/v1", tags=["investigation"])


# ── Health Check ──────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root() -> dict:
    """Health check — confirms the API is running."""
    return {
        "service": "AI Cyber Discovery Engine",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}
