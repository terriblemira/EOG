# backend/app.py
import asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from stream import get_stream_inlet

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Game Glasses")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/calibrate", response_class=HTMLResponse)
def calibrate(request: Request):
    print("calibrate here")
    return templates.TemplateResponse("calibrate.html", {"request": request})

@app.get("/games", response_class=HTMLResponse)
def games(request: Request):
    return templates.TemplateResponse("games.html", {"request": request})

# ---- Shared inlet + broadcaster ----
inlet_task = None
inlet_queue = asyncio.Queue(maxsize=1024)  # samples for all clients
clients = set()
inlet_lock = asyncio.Lock()

async def inlet_pump():
    """Background task: read from LSL and push samples to inlet_queue."""
    inlet = await asyncio.to_thread(get_stream_inlet)  # blocks until found or raises
    while True:
        sample, ts = await asyncio.to_thread(inlet.pull_sample, 0.02)
        if sample is not None:
            item = {"t": ts, "v": sample}
            # drop oldest if queue is full
            if inlet_queue.full():
                try:
                    inlet_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await inlet_queue.put(item)

async def ensure_inlet_task():
    global inlet_task
    async with inlet_lock:
        if inlet_task is None or inlet_task.done():
            inlet_task = asyncio.create_task(inlet_pump())

@app.websocket("/ws/lsl")
async def lsl_ws(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        await ensure_inlet_task()
        await ws.send_text("Connected. Streaming shared LSL inlet…")
        # Each client has its own consumer loop
        while True:
            item = await inlet_queue.get()
            await ws.send_json(item)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)
