from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import engine, Base
from .routers import applications, assessments, auth, documents, metrics, policy, regulatory, scoring

# Create all tables in the database (SQLite for local, or Postgres)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Halcyon Credit - Agentic Underwriting Copilot API")

# Setup CORS for the UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(applications.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(scoring.router, prefix="/api")
app.include_router(policy.router, prefix="/api")
app.include_router(regulatory.router, prefix="/api")
app.include_router(assessments.router, prefix="/api")
app.include_router(assessments.assess_router, prefix="/api")
app.include_router(metrics.router, prefix="/api")


@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Halcyon API is running"}
