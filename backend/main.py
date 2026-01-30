#M: RUN IN Command Prompt under "+"
import asyncio
import uvicorn 
from eog_reader import EOGReader
import websockets
import collections
import app
import threading
import webbrowser
import minecraft_control
import calibration
import test
import utils

def start_appy(): 
    uvicorn.run(app.app, reload=False)


#M: async def main() function as event loop ("Motor" that lets async functions work) (Mainthread that is started(run) first (see at end of code))
async def main():
    # Run calibration; before starting EOG reader to have calibration params ready
   # appy_thread = threading.Thread(target=start_appy, daemon=True) #M: app.py (opening + running of webpage) as separate thread (running paralllel to eog_reader); "daemon=True": background thread, so automatically stops when main program is stopped
   # appy_thread.start()
    #print(f"FastAPI started") 

    #webbrowser.open("http://localhost:8000/games")
    #await asyncio.sleep(2)

 # 1. INITIALIZE PYGAME & EOG READER
    window, WIDTH, HEIGHT, font_size, font, clock = utils.init_pygame() #M: Maybe not needed. order is important!! (must be same as returned in utils.py)
    det_queue = collections.deque(maxlen=50)
    eog_thread = EOGReader(det_queue) #creating an instance of EOGReader class with det_queue as argument (used in the __init__ method (--> variable self.out_queue IS det_queue for this EOGReader instance (for eog_thread).)
    #M added:
    #await eog.connect_to_webapp() #M: verbindet & hält Verbindung zu app.py
    eog_thread.start() #M: start eog_reader (thread) with default calibration_params (default thresholds, etc.)

# 2. RUN CALIBRATION
    calibration_params = calibration.run_calibration(eog_thread, window, font, clock)
    blink_calibration_results = calibration.run_blink_calibration(eog_thread, window, font, clock, calibration_params, WIDTH, HEIGHT)
# 3. SAVE AND UPDATE CALIBRATION DATA
    saved_calibration_data = calibration.save_and_update_calib_data(eog_thread, calibration_params, blink_calibration_results)
    if saved_calibration_data:  #M: if calib was not quit with key "q" or unsuccessful:  
# 4. RUN/SKIP TEST
        test.run_test(eog_thread, window, font, clock, saved_calibration_data) #M: run main function from test



# 5. START MOUSE- & KEYBOARD-REPLACEMENT
   # if test.calib_and_test_completed:
    mouseKeyboard_thread = minecraft_control.MouseKeyboardReplacement()
    mouseKeyboard_thread.start() #M: calls run() - method in minecraft_control in MouseReplacement
    #keyboard_thread = minecraft_control.KeyBoardReplacement()
    #keyboard_thread.start() #M: calls run() - method in minecraft_control in KeyboardReplacement    
    print("> Mouse & Keyboard Replacement started")
    await asyncio.sleep(1)

    while True:#M: Keep the main function alive (don't block the event loop)
        await asyncio.sleep(1)   #M: only this function sleeps(=waits) repeatedly for 1 second(-->forever) (so "restaurant(main function) is not closing but stays open" after last line of the function), other tasks working parallelly


asyncio.run(main()) #M: creates and runs the event loop ("Heart of asyncio": responsible for coroutines, coordinates awaits, plans & executes I/O operations like await ws.send()...); loop stays active