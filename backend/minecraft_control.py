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
    def __init__():
        pyautogui.FAILSAFE = True     #M: stops when mouse moved to corner 

        #M: method run as thread
    def move_continuously(self):

        try:
            while True:

                if not eog_reader.signal.empty():
                    self.direction = eog_reader.signal.get()
                    print(f"direction {self.direction}")
                
                if self.direction == "left":
                    start_time = time.time()
                    while time.time() - start_time < config.TURNING_COOLDOWN:  #M: COOLDOWN: Ignore opposite direction signal for 0.5 seconds
                        pyautogui.moveRel(-self.speed, 0)
                        eog_reader.signal.clear()  #M: clear queue to avoid getting old signals during cooldown

                    #M: After 0.5 s: keep moving UNTIL opposite direction signal received
                    while True:
                        if not eog_reader.signal.empty(): #M: if empty moves straight to moveRel & checks again next iteration
                            self.checkingoppositedirection = eog_reader.signal.get()
                            if self.checkingoppositedirection == "right":
                                break #M: stop this inner "while True", when "right" detected
                        pyautogui.moveRel(-self.speed, 0) #M: if emppty OR not opp. direction

                elif self.direction == "right":
                    start_time = time.time()
                    while time.time() - start_time < config.TURNING_COOLDOWN:  #M: COOLDOWN: Ignore opposite direction signal for 0.5 seconds
                        pyautogui.moveRel(self.speed, 0)
                        eog_reader.signal.clear()  #M: clear queue to avoid getting old signals during cooldown

                    #After 0.5 s: keep moving UNTIL opposite direction signal received
                    while True:
                        if not eog_reader.signal.empty():
                            self.checkingoppositedirection = eog_reader.signal.get()
                            if self.checkingoppositedirection == "left":
                                break
                        pyautogui.moveRel(self.speed, 0)

                elif self.direction == "up":
                    start_time = time.time()
                    while time.time() - start_time < config.TURNING_COOLDOWN:  #M: COOLDOWN: Ignore opposite direction signal for 0.5 seconds
                        pyautogui.moveRel(0, -self.speed)
                        eog_reader.signal.clear()  #M: clear queue to avoid getting old signals during cooldown

                    #After 0.5 s: keep moving UNTIL opposite direction signal received
                    while True:
                        if not eog_reader.signal.empty():
                            self.checkingoppositedirection + eog_reader.signal.get()
                            if self.checkingoppositedirection == "down":
                                break
                        pyautogui.moveRel(0, -self.speed)

                elif self.direction == "down":
                    start_time = time.time()
                    while time.time() - start_time < config.TURNING_COOLDOWN:  #M: COOLDOWN: Ignore opposite direction signal for 0.5 seconds
                        pyautogui.moveRel(0, self.speed)
                        eog_reader.signal.clear()  #M: clear queue to avoid getting old signals during cooldown

                    #After 0.5 s: keep moving UNTIL opposite direction signal received
                    while True:
                        if not eog_reader.signal.empty():
                            self.checkingoppositedirection = eog_reader.signal.get()
                            if self.checkingoppositedirection == "up":
                                break
                        pyautogui.moveRel(0, self.speed)

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
        pyautogui.FAILSAFE

    # moving forward when double blink within 1 s
    def move_forward(self):
        try:
            while True:
                if not eog_reader.signal.empty():
                    self.direction == eog_reader.signal.get()
                if self.direction == "blink":
                    start_time = time.time
                    while 0.05 < time.time - start_time < 1: #M: look for 2nd blink within 1 s; "0.05"-cooldown so it maybe doesnt take 1. blink twice)
                        if self.direction == "blink":
                            pyautogui.keyDown('w')
                            print(f"Holding W down")
                            time.sleep(0.5) #M: runs at least 0.5 s before player can stop to avoid eventual weird stuff
                            while True:
                                if not eog_reader.signal.empty():
                                    self.direction == eog_reader.signal.get()
                                if self.direction == "blink":   
                                    start_time = time.time
                                    while 0.05 < time.time - start_time < 1: #M: look for 2nd blink within 1 s; "0.05"-cooldown so it maybe doesnt take 1. blink twice)
                                        if self.direction == "blink":
                                            pyautogui.keyUp('w')
                                            print(f"Stop W")

        except Exception as e:
            print(f"Error in getting forward signal")

            
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



