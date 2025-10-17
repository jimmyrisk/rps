"""
Health and status route handlers.
"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/healthz")
def healthz():
    """Health check endpoint"""
    return {"ok": True}