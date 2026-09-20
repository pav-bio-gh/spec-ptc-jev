"""Live race in the browser: nothing recorded, nothing staged.

    uv run --env-file <env with OPENAI_API_KEY + TYPESAFE_API_KEY> python -m examples.live
    open http://127.0.0.1:8765        # press Run

Pressing Run starts THREE real agent turns at the same moment, on the same task
(examples/browse_race.py): no speculation, spec-ptc as shipped, and our loop with Jev. Each lane
has its own live model stream and its own real headless Chromium. Every bar on the page is drawn
from an event this process emitted at the moment it happened. The page draws; it decides nothing.

The three lanes share one network connection, so a single live race is noisier than the
one-arm-at-a-time medians from `examples.browse_race`, which the page shows underneath.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from examples.browse_race import (
    ARMS,
    OUT,
    POLICY,
    ROOT_MODEL,
    BrowserWorker,
    run_arm,
    summarize,
)
from spec_ptc_jev import JevJudge

HOST, PORT = "127.0.0.1", 8765
PAGE = Path(__file__).parent / "live.html"


class Hub:
    """Fan events out to every connected page; remember the current race for late joiners."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.clients: list[queue.Queue] = []
        self.log: list[dict] = []
        self.running = False

    def publish(self, ev: dict) -> None:
        with self.lock:
            self.log.append(ev)
            clients = list(self.clients)
        for q in clients:
            q.put(ev)

    def join(self) -> tuple[queue.Queue, list[dict]]:
        q: queue.Queue = queue.Queue()
        with self.lock:
            self.clients.append(q)
            return q, list(self.log)

    def leave(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)


HUB = Hub()
_JUDGE: JevJudge | None = None


def shared_judge() -> JevJudge:
    """One judge for the life of the server, as a long-running agent process would have."""
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = JevJudge()
    return _JUDGE


def race_arm(arm: str, worker: BrowserWorker, go: threading.Barrier) -> None:
    def listener(kind: str, t: float, **data) -> None:
        HUB.publish({"arm": arm, "kind": kind, "t": t, "wall": time.time(), **data})

    def start() -> None:
        go.wait()
        listener("arm_start", t=0.0)

    record = run_arm(arm, worker, shared_judge(), listener=listener, go=start)
    listener(
        "arm_done",
        t=record["wall"],
        errors=record["errors"],
        n_early=record["n_early"],
        opens=record["n_open_calls"],
        searches=record["n_search_calls"],
        search_early=record["search_early"],
        answer=record["final_answer"] or "",
        answer_correct=record["answer_correct"],
    )


def run_race() -> None:
    with HUB.lock:
        if HUB.running:
            return
        HUB.running = True
        HUB.log.clear()
    HUB.publish(
        {"arm": "*", "kind": "race_start", "t": 0, "policy": POLICY, "root_model": ROOT_MODEL}
    )
    workers = {arm: BrowserWorker() for arm in ARMS}
    try:
        go = threading.Barrier(len(ARMS))
        threads = [
            threading.Thread(target=race_arm, args=(arm, workers[arm], go), daemon=True)
            for arm in ARMS
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=240)
    finally:
        for w in workers.values():
            try:
                w.close()
            except Exception:
                pass
        HUB.publish({"arm": "*", "kind": "race_done", "t": 0})
        with HUB.lock:
            HUB.running = False


def eval_summary() -> dict:
    try:
        d = json.loads(OUT.read_text())
        return {
            "recorded": d.get("recorded"),
            "root_model": d.get("root_model"),
            "rounds": max(sum(1 for r in d["runs"] if r["arm"] == a) for a in ARMS),
            **summarize(d["runs"]),
        }
    except Exception as e:
        return {"error": str(e)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/eval":
            self._send(200, json.dumps(eval_summary()).encode(), "application/json")
        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            q, backlog = HUB.join()
            try:
                for ev in backlog:
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        ev = q.get(timeout=15)
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                HUB.leave(q)
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/run":
            self._send(404, b"not found", "text/plain")
        elif self.headers.get("Host", "").split(":")[0] not in ("127.0.0.1", "localhost"):
            self._send(403, b"local only", "text/plain")
        else:
            with HUB.lock:
                busy = HUB.running
            if not busy:
                threading.Thread(target=run_race, daemon=True).start()
            self._send(202, json.dumps({"started": not busy}).encode(), "application/json")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"live race: http://{HOST}:{PORT}   (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
