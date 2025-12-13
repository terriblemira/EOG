#M: before running: run app.py to open webpage
#M: RUN IN Command Prompt under "+"
import asyncio
import uvicorn 
from eog_reader import EOGReader
import websockets
import test
import collections
import app
import threading
import webbrowser
import minecraft_control

def start_appy(): 
    uvicorn.run(app.app, reload=False)

def start_mouse_replacementpy():
    minecraft_control.MouseMover()


    


#M: async def main() function as event loop ("Motor" that lets async functions work) (Mainthread that is started(run) first (see at end of code))
async def main():
    # Run calibration; before starting EOG reader to have calibration params ready
    appy_thread = threading.Thread(target=start_appy, daemon=True) #M: app.py (opening + running of webpage) as separate thread (running paralllel to eog_reader); "daemon=True": background thread, so automatically stops when main program is stopped
    appy_thread.start()
    print(f"FastAPI started") 
    mouse_replacementpy_thread = threading.Thread(target=start_mouse_replacementpy, daemon=True)
    await asyncio.sleep(1)

    webbrowser.open("http://localhost:8000/games")

    await asyncio.sleep(10)

    # # Initialize EOG reader 
    # det_queue = collections.deque(maxlen=50)
    # eog = EOGReader(det_queue)
    # eog.eventLoop = asyncio.get_event_loop() #M: Websocket Setup
    # #await eog.connect_to_webapp() #M: connect & holding connection to app.py (from eog_reader.py)
    # eog.start() #M: start eog_reader (thread)

    test.main() #M: run main function from test

    while True:#M: Keep the main function alive (don't block the event loop)
        await asyncio.sleep(1)   #M: only this function sleeps(=waits) repeatedly for 1 second(-->forever) (so "restaurant(main function) is not closing but stays open" after last line of the function), other tasks working parallelly


asyncio.run(main()) #M: creates and runs the event loop ("Heart of asyncio": responsible for coroutines, coordinates awaits, plans & executes I/O operations like await ws.send()...); loop stays active