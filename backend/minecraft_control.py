#creates continuous turning in direction of eye movement signal until looking in another direction + presses 'w' if double-blinking + stops 'w' if another blink
#TODO: check how often signal queue is updated (every 0.1s?) - if too slow, change in eog_reader.py
#TODO: (maybe in another class bc mouse movements parallel to moving forward): keep W-key pressed when double-blink detected (2 blinks within 1 s), regardless of direction & jump signals (not as elif), so that during moving forward turning & jumps can happen simultaneously: just stop w other double blink
#TODO: get threadings right!! (turning and moving forward at same time parallel, as well as eog_reader thread)
import pyautogui
import eog_reader
import threading
import time
import config

class MouseKeyboardReplacement(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.speed = 10
        self.last_blink_time = None
        self.wPressed = False
        self.leftPressed = False
        self.rightPressed = False
        self.upPressed = False
        self.downPressed = False
        self.direction = None
        pyautogui.FAILSAFE = True     #M: stops when mouse moved to corner 
        print(f"Class MouseKeyboardReplacement started as thread")
                    # DEBUG
        print(f"MINECRAFT_CONTROL: Queue type: {type(eog_reader.signal)}")
        print(f"Queue ID: {id(eog_reader.signal)}")
        print(f"Has clear: {hasattr(eog_reader.signal, 'clear')}")
        print(f"EOGReader ID: {id(eog_reader)}")

    def run(self):
        while self.running:

            if not eog_reader.signal.empty():
                self.direction = eog_reader.signal.get()
                print(f"minecraft_control: direction {self.direction}")

                if self.direction == "blink":
                    self.move_forward()
                    self.direction = None  # reset direction after processing blink, not just queue, so can detect other directions
                else:
                    self.move_continuously()
            time.sleep(0.01) # for CPU not jumping to 100%, should not b interfering with signal income or sth bc very short


        #M: method run as thread
    def move_continuously(self):
                
        if self.direction == "left":
            if self.rightPressed:
                pyautogui.keyUp('right')
                self.rightPressed = False
            if not self.leftPressed:
                pyautogui.keyDown('left') # string for left-arrow-key
                self.leftPressed = True
                start_time = time.time()
                while time.time() - start_time < config.TURNING_COOLDOWN:  #M: COOLDOWN: Ignore opposite direction signal for 0.5 seconds
                    #pyautogui.moveRel(-self.speed, 0) #M: moves 1 time by 10 pixels --> has to be IN while-loop, not like keyDown
                    eog_reader.signal.clear()  #M: clear queue to avoid getting old signals during cooldown
                    time.sleep(0.01) #M: to avoid getting cleared millions of times (CPU 1000) ## could cause lagging-problems if working w moveRel
                
                #M: After 0.5 s(Turning_cooldown): keep moving UNTIL opposite direction signal received
 #               while True:
 #                   if not eog_reader.signal.empty(): #M: if empty moves straight to keyUp & checks again next iteration
 #                       self.checkingoppositedirection = eog_reader.signal.get()
 #                       if self.checkingoppositedirection == "right":
 #                           pyautogui.keyUp('left')
 #                           break #M: stop this inner "while True" if "right" detected
 #                   #pyautogui.moveRel(-self.speed, 0)
 #                   time.sleep(0.01) # if empty: jumps here + restarts while-True-checking-loop (!! evtl. PROBLEM bc nothing being detected within those 0.01 s?)

        elif self.direction == "right":
            if self.leftPressed:
                pyautogui.keyUp('left')
                self.leftPressed = False
            if not self.rightPressed:
                pyautogui.keyDown('right')
                self.rightPressed = True
                start_time = time.time()
                while time.time() - start_time < config.TURNING_COOLDOWN:  #M: COOLDOWN: Ignore direction signal for 0.5 seconds
                    #pyautogui.moveRel(self.speed, 0)
                    eog_reader.signal.clear()  #M: clear queue to avoid getting old signals during cooldown
                    time.sleep(0.01)

            # #After 0.5 s: keep moving UNTIL opposite direction signal received
            # while True:
            #     if not eog_reader.signal.empty():
            #         self.checkingoppositedirection = eog_reader.signal.get()
            #         if self.checkingoppositedirection == "left":
            #             pyautogui.keyUp('right')
            #             break
            #     # pyautogui.moveRel(self.speed, 0)
            #     time.sleep(0.01)

        elif self.direction == "up":
            if self.downPressed:
                pyautogui.keyUp('down')
                self.downPressed = False
            if not self.upPressed:
                pyautogui.keyDown('up')
                self.upPressed = True
                start_time = time.time()
                while time.time() - start_time < config.TURNING_COOLDOWN:  #M: COOLDOWN: Ignore opposite direction signal for 0.5 seconds
                    eog_reader.signal.clear()  #M: clear queue to avoid getting old signals during cooldown
                    #pyautogui.moveRel(0, -self.speed)
                    time.sleep(0.01)

            # #After 0.5 s: keep moving UNTIL opposite direction signal received
            # while True:
            #     if not eog_reader.signal.empty():
            #         self.checkingoppositedirection + eog_reader.signal.get()
            #         if self.checkingoppositedirection == "down":
            #             pyautogui.keyUp('up')
            #             break
            #     #pyautogui.moveRel(0, -self.speed)
            #     time.sleep(0.01)

        elif self.direction == "down":
            if self.upPressed:
                pyautogui.keyUp('up')
                self.upPressed = False
            if not self.downPressed:
                pyautogui.keyDown('down')
                self.downPressed = True
                start_time = time.time()
                while time.time() - start_time < config.TURNING_COOLDOWN:  #M: COOLDOWN: Ignore opposite direction signal for 0.5 seconds
                    #pyautogui.moveRel(0, self.speed)
                    eog_reader.signal.clear()  #M: clear queue to avoid getting old signals during cooldown
                    time.sleep(0.01) # remove if working with moveRel

            # #After 0.5 s: keep moving UNTIL opposite direction signal received
            # while True:
            #     if not eog_reader.signal.empty():
            #         self.checkingoppositedirection = eog_reader.signal.get()
            #         if self.checkingoppositedirection == "up":
            #             pyautogui.keyUp('down')
            #             break
            #     time.sleep(0.01)
            #     #pyautogui.moveRel(0, self.speed)


    # moving forward if double blink within 1 s
    def move_forward(self):
        current_time = time.time() # time-stamp as soon as blinked

        #M: in case last_blink_time already existing (check for valid double blink):
        if self.last_blink_time and (current_time - self.last_blink_time < 1): # First one i.o. to say that if last_b_t is None (as in init), skip
            if self.wPressed:
                pyautogui.keyUp('w')
                print(f'W released')
                self.wPressed = False
            
            else:
                pyautogui.keyDown('w')
                print(f'W pressed')
                self.wPressed = True
            
            self.last_blink_time = None #M: reset after double blink

        #M: if that was first blink within last second
        else:
            self.last_blink_time = current_time
            print(f'{self.last_blink_time}: Attentive for second blink')


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

# class KeyBoardReplacement(threading.Thread):
    
#     def __init__(self):
#         super().__init__()
#         self.running = True
#         pyautogui.FAILSAFE

#     def run(self):
#         while self.running:
#             if not eog_reader.signal.empty():
#                 self.move_forward()
#             time.sleep(0.01) # for CPU not jumping to 100%, should not b interfering with signal income or sth bc very short

    

            
if __name__ == "__main__":
#    keyBoardReplacement = KeyBoardReplacement()
    mouseKeyboardReplacement = MouseKeyboardReplacement()


    #Start thread

#M Keep the main thread alive to allow continuous movement
    # try:
    #     while True:
    #         time.sleep(1)  # Keep the main thread alive
    # except KeyboardInterrupt:
    #     MouseReplacement.stop_move_continuously()
    #     print("Program terminated.")



