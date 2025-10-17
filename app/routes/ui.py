"""
UI route handlers for web interface.
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from datetime import datetime, timezone
import os

from app.config import get_metrics_since_date

# Setup templates
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Configuration
DEFAULT_UI_API_BASE = (os.getenv("UI_CLIENT_API_BASE") or "").rstrip("/")
GRAFANA_URL = os.getenv("GRAFANA_DASHBOARD_URL", "")
ASSET_VERSION = os.getenv("ASSET_VERSION")
if not ASSET_VERSION:
    ASSET_VERSION = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

router = APIRouter()

# ...existing imports...

def _render_lite_template(request: Request, template_name: str, api_base: str | None) -> HTMLResponse:
    resolved_base = (api_base if api_base is not None else DEFAULT_UI_API_BASE) or ""
    resolved_base = resolved_base.rstrip("/")
    context = {
        "request": request,
        "api_base": resolved_base,
        "grafana_url": GRAFANA_URL,
        "asset_version": ASSET_VERSION,
        "metrics_reset": get_metrics_since_date(),
    }
    resp = templates.TemplateResponse(template_name, context)
    # Aggressive cache prevention for Safari and other stubborn browsers
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    # ETag based on asset version to force Safari to check for updates
    resp.headers["ETag"] = f'"{ASSET_VERSION}"'
    return resp

@router.get("/", response_class=HTMLResponse)
@router.get("/ui-lite", response_class=HTMLResponse)
def serve_lite_ui(request: Request, api_base: str = None):
    """Simple FastAPI-native UI for playing the game without Streamlit."""
    return _render_lite_template(request, "rps_lite.html", api_base)

@router.get("/ui-lite-debug", response_class=HTMLResponse)
def serve_lite_ui_debug(request: Request, api_base: str = None):
    """Expose the lite UI with developer tooling enabled."""
    return _render_lite_template(request, "rps_lite_debug.html", api_base)