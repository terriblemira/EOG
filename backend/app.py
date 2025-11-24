# backend/app.py
#M: Webbrowser: Forbidden to browser to directly coonect to backend (CORS policy), so app.py acts as intermediary between browser and backend (eog_reader.py)
#M: If separate start (from main.py): TYPE in conda (under New Terminal --> Command prompt): cd C:\Users\mirad\GIT\EOG-1\backend --> ENTER --> python -m uvicorn app:app --reload
#M: to check if WS connection is established: open console/network with f12 once webpage open
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates # fastAPI uses Jinja2 for rendering(=(anzeigen/darstellen/filling in data in)) Templates(=Vorlagen)
from stream import get_stream_inlet, has_lsl_stream
from signal_interpret import SignalInterpreter, CalibrationConfig   # <-- add CalibrationConfig
from calibration_placeh import CalibrationSession
import eog_reader
import main_calib_and_pyg as main_c_and_p
from queue import Queue

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Game Glasses")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get("/printingMovementFromReader", response_class=HTMLResponse)
def printingMovement(request: Request):
    return templates.TemplateResponse("printingMovementFromReader.html", {"request": request})

#Mira bc she doesn't know where to put this code, adding it here for now
@app.websocket("/wsMovement")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("EOG_reader connected to /wsMovement")

    try:
        counter = 0
        while True: #M: to keep connection alive, attentive for incoming signals from eog_reader.py
            counter += 1
            if counter % 10 == 0:  # every 1 second
                 is_empty = eog_reader.signal.empty() #DEBUG
                 queue_size = eog_reader.signal.qsize() #DEBUG
                 #print(f"Loop {counter} running: Queue empty={is_empty}, size={queue_size}")  # DEBUG
                 print(f"signal queue contents (directions): {list(eog_reader.signal.queue)}")  # DEBUG
            
            if not eog_reader.signal.empty():
                #and main_c_and_p.is_calib_running == True:
                direction = eog_reader.signal.get()
                print(f"direction received: {direction}")
                if direction == "left":
                    print(f"{direction}: Sending 'Left-direction received' message to JavaScript")
                    await websocket.send_text("Left-command received")
                elif direction == "right":
                    print(f"{direction}: Sending 'Right-direction received' message to JavaScript")  # DEBUG)
                    await websocket.send_text("Right-command received")
                elif direction == "up":
                    print(f"{direction}: Sending 'Up-direction received' message to JavaScript")
                    await websocket.send_text("Up-command received")
                elif direction == "down":
                    print(f"{direction}: Sending 'Down-direction received' message to JavaScript")
                    await websocket.send_text("Down-command received")
                elif direction == "blink":
                    print(f"{direction}: Sending 'BLINK' message to JavaScript")
                    await websocket.send_text("BLINK")
            else:
                await websocket.send_text("empty signal queue")
            await asyncio.sleep(0.1) # M: eventually remove/make shorter
    except Exception as e_ws_to_JavaScript:
        print(f"Error in websocket endpoint: {e_ws_to_JavaScript}")
    except WebSocketDisconnect:
        print("WebSocket to EOG disconnected")


# ---------- Pages ----------
@app.get("/", response_class=HTMLResponse)
def signal_check(request: Request): #request = "note that client brings in" (f.ex. header, cookies, 
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

@app.get("/pong", response_class=HTMLResponse)
def pong(request: Request):
    return templates.TemplateResponse("pong.html", {"request": request})


# # ---------- API for signal presence ----------
# @app.get("/api/lsl/status")
# async def lsl_status(
#     name: str = Query("Explore_8441_ExG"),
#     timeout: float = Query(0.8)
# ):
#     ok = await asyncio.to_thread(has_lsl_stream, name, timeout)
#     return JSONResponse({"ok": ok, "name": name})


# # ---------- Shared LSL inlet + broadcaster ----------
# _clients_lsl: set[WebSocket] = set()
# _clients_cmd: set[WebSocket] = set()
# _pump_task: asyncio.Task | None = None
# _pump_lock = asyncio.Lock()

# CHUNK_MAX_SAMPLES = 32
# PULL_TIMEOUT_SEC = 0.05
# TARGET_SEND_HZ = 20

# # Interpreter with initial defaults
# _interpreter = SignalInterpreter(
#     CalibrationConfig(ema_alpha=0.2, cooldown_ms=200)
# )

# async def _lsl_pump():
#     loop = asyncio.get_event_loop()
#     last_send = 0.0
#     while True:
#         inlet = None
#         try:
#             inlet = await asyncio.to_thread(get_stream_inlet)
#             _interpreter.reset()
#         except Exception:
#             await asyncio.sleep(0.5)
#             continue

#         for ws in list(_clients_lsl | _clients_cmd):
#             try:
#                 await ws.send_text("LSL:connected")
#             except Exception:
#                 pass

#         try:
#             while True:
#                 chunk, ts = await asyncio.to_thread(
#                     inlet.pull_chunk, PULL_TIMEOUT_SEC, CHUNK_MAX_SAMPLES
#                 )
#                 if chunk and ts:
#                     # broadcast to /ws/lsl
#                     now = loop.time()
#                     if _clients_lsl:
#                         min_interval = 1.0 / TARGET_SEND_HZ
#                         if now - last_send < min_interval:
#                             await asyncio.sleep(min_interval - (now - last_send))
#                         last_send = loop.time()
#                         msg = {"t": ts, "v": chunk}
#                         dead = []
#                         for ws in list(_clients_lsl):
#                             try:
#                                 await ws.send_json(msg)
#                             except Exception:
#                                 dead.append(ws)
#                         for ws in dead:
#                             _clients_lsl.discard(ws)

#                     # run interpreter → maybe emit command to /ws/cmd
#                     cmd = _interpreter.process_chunk(chunk)
#                     if cmd and _clients_cmd:
#                         dead2 = []
#                         for ws in list(_clients_cmd):
#                             try:
#                                 await ws.send_json({"cmd": cmd})
#                             except Exception:
#                                 dead2.append(ws)
#                         for ws in dead2:
#                             _clients_cmd.discard(ws)
#                 else:
#                     await asyncio.sleep(0.005)
#         except Exception:
#             await asyncio.sleep(0.5)
#             continue

# async def _ensure_pump():
#     global _pump_task
#     async with _pump_lock:
#         if _pump_task is None or _pump_task.done():
#             _pump_task = asyncio.create_task(_lsl_pump())


# # ---------- WebSocket endpoints ----------
# @app.websocket("/ws/lsl")
# async def lsl_ws(ws: WebSocket):
#     await ws.accept()
#     _clients_lsl.add(ws)
#     await _ensure_pump()
#     try:
#         await ws.send_text("WS:ready")
#         while True:
#             await asyncio.sleep(60)
#     except WebSocketDisconnect:
#         pass
#     finally:
#         _clients_lsl.discard(ws)


# _cal_session: CalibrationSession | None = None
# _cal_phase: str | None = None

# @app.websocket("/ws/cal")
# async def cal_ws(ws: WebSocket):
#     global _cal_session, _cal_phase
#     await ws.accept()
#     await _ensure_pump()
#     if _cal_session is None:
#         _cal_session = CalibrationSession()
#     await ws.send_json({"status":"ready"})

#     try:
#         while True:
#             msg = await ws.receive_json()
#             typ = msg.get("type")

#             if typ == "start":
#                 _cal_session = CalibrationSession()
#                 _cal_phase = None
#                 await ws.send_json({"status": "started"})

#             elif typ == "phase":
#                 name = msg.get("name")
#                 duration = float(msg.get("duration_ms", 2000)) / 1000.0
#                 _cal_phase = name
#                 await ws.send_json({"status":"phase_started","name":name})

#                 inlet = await asyncio.to_thread(get_stream_inlet)
#                 import time
#                 t0 = time.time()
#                 while time.time() - t0 < duration:
#                     chunk, ts = await asyncio.to_thread(inlet.pull_chunk, 0.05, 64)
#                     if chunk:
#                         _cal_session.feed_chunk(name, chunk)
#                     await asyncio.sleep(0.005)

#                 await ws.send_json({"status":"phase_done","name":name})
#                 _cal_phase = None

#             elif typ == "finish":
#                 cfg = _cal_session.compute_config()
#                 _interpreter.update_config(cfg)
#                 await ws.send_json({"status":"finished","config":cfg.__dict__})

#             else:
#                 await ws.send_json({"status":"error","detail":"unknown message"})
#     except WebSocketDisconnect:
#         pass


# @app.websocket("/ws/cmd")
# async def cmd_ws(ws: WebSocket):
#     await ws.accept()
#     _clients_cmd.add(ws)
#     await _ensure_pump()
#     try:
#         await ws.send_text("WS:cmd-ready")
#         while True:
#             await asyncio.sleep(60)
#     except WebSocketDisconnect:
#         pass
#     finally:
#         _clients_cmd.discard(ws)
