"""Local review server: 127.0.0.1 only, serves the coded items and records answers.

``key.json`` is never served -- the browser sees clip codes and, for the boundary task,
the boundaries under review. Answers go to an append-only ``answers.jsonl``; a corrected
answer is a later line, so nothing is ever overwritten.
"""

# No `from __future__ import annotations` here: FastAPI resolves the `Request` annotation
# at runtime, and a string annotation naming a local import turns it into a query parameter.

import json
import re
import threading
from datetime import datetime
from pathlib import Path

__all__ = ["create_app", "serve"]

_STATIC = Path(__file__).with_name("static")
_CLIP_NAME = re.compile(r"^[A-Z0-9]{6}(_ref|_ctx)?\.(wav|png)$")


def create_app(review_dir: Path):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse

    from .score import latest_answers

    review_dir = Path(review_dir)
    answers_path = review_dir / "answers.jsonl"

    def load_items() -> dict:
        # Read on every request: a refreshed or extended package shows up on reload,
        # without restarting the server.
        return json.loads((review_dir / "items.json").read_text(encoding="utf-8"))
    lock = threading.Lock()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/items")
    def get_items():
        return load_items()

    @app.get("/api/answers")
    def get_answers():
        return latest_answers(answers_path)

    @app.get("/clip/{name}")
    def clip(name: str):
        if not _CLIP_NAME.match(name):
            raise HTTPException(404)
        path = review_dir / "clips" / name
        if not path.exists():
            raise HTTPException(404)
        # no-cache: a redrawn spectrogram keeps its name, and a stale cached copy would
        # quietly show the old picture
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    @app.post("/api/answer")
    async def post_answer(request: Request):
        body = await request.json()
        task, code, answer = body.get("task"), body.get("id"), body.get("answer")
        valid = {t: {x["id"] for x in v} for t, v in load_items()["tasks"].items()}
        if task not in valid or code not in valid[task] or not isinstance(answer, dict):
            raise HTTPException(400, "unknown task or item")
        rec = {"task": task, "id": code, "answer": answer,
               "ts": f"{datetime.now():%Y-%m-%dT%H:%M:%S}"}
        with lock, answers_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"ok": True}

    return app


def _open_when_listening(url: str, port: int, timeout_s: float = 60.0) -> None:
    """Open the browser only once the port accepts connections. A fixed delay could fire
    before the server was up and show the user a 'connection refused' page."""
    import socket
    import time
    import webbrowser

    t_end = time.time() + timeout_s
    while time.time() < t_end:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
        except OSError:
            time.sleep(0.2)
            continue
        print(f"[review] 介面已就緒：{url}", flush=True)
        webbrowser.open(url)
        return


def serve(review_dir: Path, port: int = 8765, open_browser: bool = True) -> None:
    import socket

    import uvicorn

    url = f"http://127.0.0.1:{port}"
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            print(f"[review] 埠 {port} 已經有程式在用（可能介面已經開著）：直接打開 {url}",
                  flush=True)
            return
    print("[review] 啟動中……這個視窗請保持開著，關掉就會停止。答案每按一次就存一次。",
          flush=True)
    if open_browser:
        threading.Thread(target=_open_when_listening, args=(url, port), daemon=True).start()
    uvicorn.run(create_app(review_dir), host="127.0.0.1", port=port, log_level="warning")
