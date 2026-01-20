#creates continuous movements in direction of eye movement signal until looking in another direction
#TODO: check how often signal queue is updated (every 0.1s?) - if too slow, change in eog_reader.py
#TODO: (maybe in another class bc mouse movements parallel to moving forward): keep W-key pressed when double-blink detected (2 blinks within 1 s), regardless of direction & jump signals (not as elif), so that during moving forward turning & jumps can happen simultaneously: just stop w other double blink
#TODO: get threadings right!! (turning and moving forward at same time parallel, as well as eog_reader thread)
import pyautogui
import eog_reader
import threading
import time
import config

class MouseReplacement(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.speed = 10
        pyautogui.FAILSAFE = True     #M: stops when mouse moved to corner 
        print(f"Class MouseReplacement started as thread")

    def run(self):
        while self.running:
            if not eog_reader.signal.empty():
                self.move_continuously()
            time.sleep(0.01) # for CPU not jumping to 100%, should not b interfering with signal income or sth bc very short

        #M: method run as thread
    def move_continuously(self):

        try:
            while True:

                if not eog_reader.signal.empty():
                    self.direction = eog_reader.signal.get()
                    print(f"minecraft_control: direction {self.direction}")
                
                if self.direction == "left":
                    pyautogui.keyDown('left') # string for left-arrow-key
                    start_time = time.time()
                    while time.time() - start_time < config.TURNING_COOLDOWN:  #M: COOLDOWN: Ignore opposite direction signal for 0.5 seconds
                        #pyautogui.moveRel(-self.speed, 0) #M: moves 1 time by 10 pixels --> has to be IN while-loop, not like keyDown
                        eog_reader.signal.clear()  #M: clear queue to avoid getting old signals during cooldown
                        time.sleep(0.01) #M: to avoid getting cleared millions of times (CPU 1000) ## could cause lagging-problems if working w moveRel
                        
                    #M: After 0.5 s(Turning_cooldown): keep moving UNTIL opposite direction signal received
                    while True:
                        if not eog_reader.signal.empty(): #M: if empty moves straight to keyUp & checks again next iteration
                            self.checkingoppositedirection = eog_reader.signal.get()
                            if self.checkingoppositedirection == "right":
                                pyautogui.keyUp('left')
                                break #M: stop this inner "while True" if "right" detected
                        #pyautogui.moveRel(-self.speed, 0)
                        time.sleep(0.01) # if empty: jumps here + restarts while-True-checking-loop (!! evtl. PROBLEM bc nothing being detected within those 0.01 s?)

                elif self.direction == "right":
                    pyautogui.keyDown('right')
                    start_time = time.time()
                    while time.time() - start_time < config.TURNING_COOLDOWN:  #M: COOLDOWN: Ignore opposite direction signal for 0.5 seconds
                        #pyautogui.moveRel(self.speed, 0)
                        eog_reader.signal.clear()  #M: clear queue to avoid getting old signals during cooldown
                        time.sleep(0.01)

                    #After 0.5 s: keep moving UNTIL opposite direction signal received
                    while True:
                        if not eog_reader.signal.empty():
                            self.checkingoppositedirection = eog_reader.signal.get()
                            if self.checkingoppositedirection == "left":
                                pyautogui.keyUp('right')
                                break
                        # pyautogui.moveRel(self.speed, 0)
                        time.sleep(0.01)

                elif self.direction == "up":
                    pyautogui.keyDown('up')
                    start_time = time.time()
                    while time.time() - start_time < config.TURNING_COOLDOWN:  #M: COOLDOWN: Ignore opposite direction signal for 0.5 seconds
                        eog_reader.signal.clear()  #M: clear queue to avoid getting old signals during cooldown
                        #pyautogui.moveRel(0, -self.speed)
                        time.sleep(0.01)
                    #After 0.5 s: keep moving UNTIL opposite direction signal received
                    while True:
                        if not eog_reader.signal.empty():
                            self.checkingoppositedirection + eog_reader.signal.get()
                            if self.checkingoppositedirection == "down":
                                pyautogui.keyUp('up')
                                break
                        #pyautogui.moveRel(0, -self.speed)
                        time.sleep(0.01)

                elif self.direction == "down":
                    pyautogui.keyDown('down')
                    start_time = time.time()
                    while time.time() - start_time < config.TURNING_COOLDOWN:  #M: COOLDOWN: Ignore opposite direction signal for 0.5 seconds
                        #pyautogui.moveRel(0, self.speed)
                        eog_reader.signal.clear()  #M: clear queue to avoid getting old signals during cooldown
                        time.sleep(0.01) # remove if working with moveRel

                    #After 0.5 s: keep moving UNTIL opposite direction signal received
                    while True:
                        if not eog_reader.signal.empty():
                            self.checkingoppositedirection = eog_reader.signal.get()
                            if self.checkingoppositedirection == "up":
                                pyautogui.keyUp('down')
                                break
                        time.sleep(0.01)
                        #pyautogui.moveRel(0, self.speed)

        except Exception as e:
            print(f"Error in move_continuously: {e}")  

    #     try:
    #          counter = 0
    #          while self.running:
    #             counter += 1

    #             #M: Get new direction signal every 0.1 s (10 iterations of 0.01s sleep)
    #             #TODO: make sure next signal in queue that gets is current one, not old one (clear queue before getting new one?)
    #             if counter % 10 == 0:
    #                 if not eog_reader.signal.empty():
    #                     self.direction = eog_reader.signal.get()
    #                     print(f"direction {self.direction}")
                
    #             #M: not indented as much bc: Move EVERY iteration, not just % 10 == 0 (every 10th)
    #             if self.direction == "left":
    #                 pyautogui.moveRel(-self.speed, 0)
    #             elif self.direction == "right":
    #                 pyautogui.moveRel(self.speed, 0)
    #             elif self.direction == "up":
    #                 pyautogui.moveRel(0, -self.speed)
    #             elif self.direction == "down":
    #                 pyautogui.moveRel(0, self.speed)

    #             time.sleep(0.01)  # Adjust the sleep time for smoother or faster movement

    #     except Exception as e:
    #         print(f"Error in move_continuously: {e}")  

        
    # def initiate_move_continuously(self):
    #     self.running = True

    #     if not eog_reader.signal.empty():
    #         self.direction = eog_reader.signal.get()

    #     self.mouse_thread = threading.Thread(target=self.move_continuously, daemon=True) #M: run function move_cont. as thread (like while True: move_cont() but not blocking loop)
    #     self.mouse_thread.start()
    #     print(f"Started moving {self.direction}")

    # def stop_move_continuously(self):
    #     if self.running:
    #         self.running = False
    #         if self.mouse_thread: #M: if thread is alive
    #             self.mouse_thread.join() #M: stops main thread (from "signaling mousethread to stop" until "mouse thread actually stops")
    #     print("Mouse stopped moving, main thread continuing now")

class KeyBoardReplacement(threading.Thread):
    
    def __init__(self):
        super().__init__()
        self.running = True
        pyautogui.FAILSAFE

    def run(self):
        while self.running:
            if not eog_reader.signal.empty():
                self.move_forward()
            time.sleep(0.01) # for CPU not jumping to 100%, should not b interfering with signal income or sth bc very short

    # moving forward when double blink within 1 s
    def move_forward(self):
        try:
            while True:
                if not eog_reader.signal.empty():
                    self.direction = eog_reader.signal.get()
                    if self.direction == "blink":
                        eog_reader.signal.clear()
                        time.sleep(0.05) #M: "0.05"-cooldown so it maybe doesnt take 1. blink twice
                        start_time = time.time()
                        print(f'blink-timer started')

                        second_blink = False
                        while time.time() - start_time < 1: #M: look for 2nd blink within 1 (+ 0.05) s; )
                            if not eog_reader.signal.empty():
                                self.direction = eog_reader.signal.get()
                                if self.direction == "blink":
                                    second_blink = True
                                    pyautogui.keyDown('w')
                                    print(f"Start W")
                                    eog_reader.signal.clear()  #M: clear queue to avoid getting old signals during cooldown
                                    time.sleep(0.5) #M: runs at least 0.5 s before player can stop moving forward to avoid eventual weird stuff  

                                    #M: now wait on double blink to stop
                                    while True:
                                        if not eog_reader.signal.empty():
                                            self.direction = eog_reader.signal.get()
                                            if self.direction == "blink":
                                                time.sleep(0.05)
                                                start_time = time.time()
                                                print(f'blink-stopping timer started')
                                                eog_reader.signal.clear()

                                                second_stop_blink = False
                                                while time.time() - start_time < 1: #M: look for 2nd blink within 1 s; "0.05"-cooldown so it maybe doesnt take 1. blink twice
                                                    if not eog_reader.signal.empty():
                                                        self.direction = eog_reader.signal.get()
                                                        if self.direction == "blink":
                                                            second_stop_blink = True
                                                            pyautogui.keyUp('w')
                                                            print(f"Stop W")
                                                            eog_reader.signal.clear()  #M: clear queue to avoid getting old signals during cooldown
                                                    time.sleep(0.01)
                                                if second_stop_blink:
                                                    break
                                                else:
                                                    eog_reader.signal.clear()
                                        time.sleep(0.01)

                            time.sleep(0.01)

                        if not second_blink:
                            eog_reader.signal.clear()

                time.sleep(0.01)
                
        except Exception as e:
            print(f"Error in getting forward signal: {e}")

            
if __name__ == "__main__":
    keyBoardReplacement = KeyBoardReplacement()
    mouseReplacement = MouseReplacement()


    #Start thread

#M Keep the main thread alive to allow continuous movement
    # try:
    #     while True:
    #         time.sleep(1)  # Keep the main thread alive
    # except KeyboardInterrupt:
    #     MouseReplacement.stop_move_continuously()
    #     print("Program terminated.")



