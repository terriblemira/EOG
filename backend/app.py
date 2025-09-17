# backend/app.py
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from stream import get_stream_inlet, has_lsl_stream
from signal_interpret import SignalInterpreter

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Game Glasses")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ---------- Pages ----------
@app.get("/", response_class=HTMLResponse)
def signal_check(request: Request):
    return templates.TemplateResponse("signal.html", {"request": request})

@app.get("/index", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/calibrate", response_class=HTMLResponse)
def calibrate(request: Request):
    print("calibrate here")
    return templates.TemplateResponse("calibrate.html", {"request": request})

@app.get("/games", response_class=HTMLResponse)
def games(request: Request):
    return templates.TemplateResponse("games.html", {"request": request})

@app.get("/snake", response_class=HTMLResponse)
def snake(request: Request):
    return templates.TemplateResponse("snake.html", {"request": request})

# ---------- API for signal presence ----------
@app.get("/api/lsl/status")
async def lsl_status(
    name: str = Query("Explore_8441_ExG"),
    timeout: float = Query(0.8)
):
    ok = await asyncio.to_thread(has_lsl_stream, name, timeout)
    return JSONResponse({"ok": ok, "name": name})

# ---------- Shared LSL inlet + broadcaster ----------
_clients_lsl: set[WebSocket] = set()
_clients_cmd: set[WebSocket] = set()
_pump_task: asyncio.Task | None = None
_pump_lock = asyncio.Lock()

# Smooth streaming
CHUNK_MAX_SAMPLES = 32
PULL_TIMEOUT_SEC = 0.05
TARGET_SEND_HZ = 20

# Interpreter (adjust thresholds as needed)
_interpreter = SignalInterpreter(
    ema_alpha=0.2,
    thresh_h=0.02,
    thresh_v=0.02,
    cooldown_ms=200
)

async def _lsl_pump():
    """
    Background task: connect to LSL inlet; broadcast samples to /ws/lsl
    and interpreted commands to /ws/cmd.
    """
    loop = asyncio.get_event_loop()
    last_send = 0.0
    while True:
        inlet = None
        try:
            inlet = await asyncio.to_thread(get_stream_inlet)
            _interpreter.reset()
        except Exception:
            await asyncio.sleep(0.5)
            continue

        for ws in list(_clients_lsl | _clients_cmd):
            try:
                await ws.send_text("LSL:connected")
            except Exception:
                pass

        try:
            while True:
                chunk, ts = await asyncio.to_thread(
                    inlet.pull_chunk, PULL_TIMEOUT_SEC, CHUNK_MAX_SAMPLES
                )
                if chunk and ts:
                    # 1) Broadcast raw chunk to /ws/lsl subscribers (optional, keeps your signal page alive)
                    now = loop.time()
                    if _clients_lsl:
                        min_interval = 1.0 / TARGET_SEND_HZ
                        if now - last_send < min_interval:
                            await asyncio.sleep(min_interval - (now - last_send))
                        last_send = loop.time()
                        msg = {"t": ts, "v": chunk}
                        dead = []
                        for ws in list(_clients_lsl):
                            try:
                                await ws.send_json(msg)
                            except Exception:
                                dead.append(ws)
                        for ws in dead:
                            _clients_lsl.discard(ws)

                    # 2) Run interpreter → maybe emit command to /ws/cmd clients
                    cmd = _interpreter.process_chunk(chunk)
                    if cmd and _clients_cmd:
                        dead2 = []
                        for ws in list(_clients_cmd):
                            try:
                                await ws.send_json({"cmd": cmd})
                            except Exception:
                                dead2.append(ws)
                        for ws in dead2:
                            _clients_cmd.discard(ws)
                else:
                    await asyncio.sleep(0.005)
        except Exception:
            await asyncio.sleep(0.5)
            continue

async def _ensure_pump():
    global _pump_task
    async with _pump_lock:
        if _pump_task is None or _pump_task.done():
            _pump_task = asyncio.create_task(_lsl_pump())

# Raw LSL stream (signal page uses this)
@app.websocket("/ws/lsl")
async def lsl_ws(ws: WebSocket):
    await ws.accept()
    _clients_lsl.add(ws)
    await _ensure_pump()
    try:
        await ws.send_text("WS:ready")
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        _clients_lsl.discard(ws)

# Commands stream (snake listens to this)
@app.websocket("/ws/cmd")
async def cmd_ws(ws: WebSocket):
    await ws.accept()
    _clients_cmd.add(ws)
    await _ensure_pump()
    try:
        await ws.send_text("WS:cmd-ready")
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        _clients_cmd.discard(ws)
