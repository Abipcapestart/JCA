"""
Two servers over one set of handlers.

`create_fastapi_app()` is what you deploy. `run_dev_server()` uses only the
standard library, so the UI is clickable on any machine with Python and no
install step — useful for demos and for CI, and it keeps the handlers honest by
making sure nothing leaks framework types.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from .handlers import ROUTES, production_providers, set_provider_factory

UI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "ui")


# ---------------------------------------------------------------------------
# FastAPI adapter (production)
# ---------------------------------------------------------------------------

def create_fastapi_app(use_production_providers: bool = False):
    """Build the FastAPI app.

        uvicorn jca_phase1.api.server:app --reload

    Requires `fastapi` and `uvicorn`.
    """
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    if use_production_providers:
        set_provider_factory(production_providers)

    app = FastAPI(title="JCA Phase 1 — Comparator & Outcome Scoping",
                  version="1.0.0",
                  description=("Phase 1 only. PICO-set generation is Phase 2 and is "
                               "not produced by this service."))

    @app.api_route("/api/{path:path}", methods=["GET", "POST"])
    async def api(path: str, request: Request) -> Response:
        route = ("GET" if request.method == "GET" else "POST", f"/api/{path}")
        handler = ROUTES.get(route)
        if handler is None:
            return JSONResponse({"error": f"no route for {route}"}, status_code=404)
        if request.method == "GET":
            payload: Dict[str, Any] = dict(request.query_params)
        else:
            try:
                payload = await request.json()
            except Exception:  # noqa: BLE001
                payload = {}
        status, body = handler(payload)
        return JSONResponse(body, status_code=status)

    if os.path.isdir(UI_DIR):
        @app.get("/")
        async def index() -> Any:
            return FileResponse(os.path.join(UI_DIR, "index.html"))

        app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")

    return app


# Import-time app for `uvicorn jca_phase1.api.server:app`
try:  # pragma: no cover - only when fastapi is installed
    app = create_fastapi_app(
        use_production_providers=os.getenv("JCA_PRODUCTION_PROVIDERS") == "1")
except Exception:  # pragma: no cover
    app = None


# ---------------------------------------------------------------------------
# Stdlib dev server (no dependencies)
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    server_version = "JCAPhase1/1.0"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, body: Dict[str, Any]) -> None:
        self._send(status, json.dumps(body, default=str).encode("utf-8"),
                   "application/json")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, b"", "text/plain")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/"):
            handler = ROUTES.get(("GET", parsed.path))
            if handler is None:
                return self._json(404, {"error": f"no route for GET {parsed.path}"})
            params = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
            status, body = handler(params)
            return self._json(status, body)
        return self._serve_static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        handler = ROUTES.get(("POST", parsed.path))
        if handler is None:
            return self._json(404, {"error": f"no route for POST {parsed.path}"})
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "request body is not valid JSON"})
        try:
            status, body = handler(payload)
        except Exception as exc:  # noqa: BLE001
            return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
        return self._json(status, body)

    def _serve_static(self, path: str) -> None:
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        full = os.path.normpath(os.path.join(UI_DIR, rel))
        if not full.startswith(os.path.normpath(UI_DIR)) or not os.path.isfile(full):
            return self._json(404, {"error": "not found"})
        ctype = {".html": "text/html; charset=utf-8", ".js": "application/javascript",
                 ".css": "text/css", ".json": "application/json",
                 ".svg": "image/svg+xml"}.get(os.path.splitext(full)[1],
                                              "application/octet-stream")
        with open(full, "rb") as f:
            self._send(200, f.read(), ctype)

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # quiet by default


def run_dev_server(host: str = "127.0.0.1", port: int = 8000,
                   use_production_providers: bool = False) -> None:
    if use_production_providers:
        set_provider_factory(production_providers)
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"JCA Phase 1 running at http://{host}:{port}")
    print(f"  UI              http://{host}:{port}/")
    print(f"  source preflight http://{host}:{port}/api/sources/preflight")
    print("  Phase 1 only — PICO-set generation is Phase 2 and is not served here.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="JCA Phase 1 dev server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--production-providers", action="store_true",
                        help="use Bedrock + Tavily + registry APIs (needs credentials)")
    args = parser.parse_args()
    run_dev_server(args.host, args.port, args.production_providers)
