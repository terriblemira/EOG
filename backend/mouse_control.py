import pyautogui
from utils import check_double_blink
import threading
import time
import config

class MouseControl(threading.Thread):

    def __init__(self, eog_reader):
        pyautogui.FAILSAFE = True     #M: stops when mouse moved to corner 
        self.eog_reader = eog_reader
        self.last_blink_time = None
        self.speed = 10
        self.wPressed = False
        self.leftPressed = False
        self.rightPressed = False
        self.upPressed = False
        self.downPressed = False
        self.direction = None
        self.left_start_time = 0
        self.right_start_time = 0
        pyautogui.FAILSAFE = True     #M: stops when mouse moved to corner 
        self._stop_event = threading.Event()
        print(f"Class MouseControl started as thread")
    
    def stop(self):
        self._stop_event.set()
    
    def run(self):
        current_time = time.time()
        while not self._stop_event.is_set():

            if not self.eog_reader.signal.empty():
                self.direction, self.timestamp = self.eog_reader.signal.get()
                print(f"MouseControl: direction {self.direction}, {self.timestamp}")
                if self.direction == "blink":
                    self.click_if_double_blink()
                else:
                    self.move_continuously()
            time.sleep(0.01) # for CPU not jumping to 100%, should not b interfering with signal income or sth bc very short


    def click_if_double_blink(self):
            # moving forward if double blink within 1.5 s
        current_time = time.time() # time-stamp as soon as blinked

        #M: in case last_blink_time already existing (check for valid double blink):
        if self.last_blink_time and (current_time - self.last_blink_time < 1.5): # First one i.o. to say that if last_b_t is None (as in init), skip

            if self.downPressed:
                self.downPressed = False
            if self.leftPressed:
                self.leftPressed = False
            if self.rightPressed:
                self.rightPressed = False
                         
            pyautogui.click()
            print(f'double-blink: clicked')
            
            self.last_blink_time = None #M: reset after double blink

        #M: if that was first blink within last second
        else:
            self.last_blink_time = current_time
            print(f'{self.last_blink_time}: Attentive for second blink')

        #M: method run as thread
    def move_continuously(self):       
        if self.direction == "left":
            if self.rightPressed:
                self.rightPressed = False
            elif not self.leftPressed:
                self.leftPressed = True
                self.left_start_time = time.time()

                while self.leftPressed == True:
                    while time.time() - self.left_start_time < config.TURNING_COOLDOWN:  #M: COOLDOWN: Ignore opposite direction signal for 0.5 seconds
                        pyautogui.moveRel(-self.speed, 0) #M: moves 1 time by 10 pixels --> has to be IN while-loop, not like keyDown
                        while not self.eog_reader.signal.empty():
                            try:
                                self.eog_reader.signal.get_nowait()  #M: clear queue to avoid getting old signals during cooldown
                            except:
                                break
                            
                        time.sleep(0.01) #M: to avoid getting cleared millions of times (CPU 1000) ## could cause lagging-problems if working w moveRel

                    pyautogui.moveRel(-self.speed, 0)
                    while not self.eog_reader.signal.empty():
                        self.checking_opposite_direction, self.timestamp = self.eog_reader.signal.get()
                        if self.checking_opposite_direction == "right":
                            self.leftPressed = False
                            self.left_start_time = 0
                            break

                        time.sleep(0.01)
                
            if self.direction == "right":
                if self.leftPressed:
                    self.leftPressed = False
                elif not self.rightPressed:
                    self.rightPressed = True
                    self.right_start_time = time.time()

                    while self.rightPressed == True:
                        while time.time() - self.right_start_time < config.TURNING_COOLDOWN:  #M: COOLDOWN: Ignore opposite direction signal for 0.5 seconds
                            pyautogui.moveRel(self.speed, 0) #M: moves 1 time by 10 pixels --> has to be IN while-loop, not like keyDown
                            while not self.eog_reader.signal.empty():
                                try:
                                    self.eog_reader.signal.get_nowait()  #M: clear queue to avoid getting old signals during cooldown
                                except:
                                    break
                                
                            time.sleep(0.01) #M: to avoid getting cleared millions of times (CPU 1000) ## could cause lagging-problems if working w moveRel

                        pyautogui.moveRel(self.speed, 0)
                        while not self.eog_reader.signal.empty():
                            self.checking_opposite_direction, self.timestamp = self.eog_reader.signal.get()
                            if self.checking_opposite_direction == "left":
                                self.rightPressed = False
                                self.right_start_time = 0
                                break

                            time.sleep(0.01)

            if self.direction == "up":
                if self.downPressed:
                    self.downPressed = False
                elif not self.upPressed:
                    self.upPressed = True
                    self.up_start_time = time.time()

                    while self.upPressed == True:
                        while time.time() - self.up_start_time < config.TURNING_COOLDOWN:  #M: COOLDOWN: Ignore opposite direction signal for 0.5 seconds
                            pyautogui.moveRel(0, -self.speed) #M: moves 1 time by 10 pixels --> has to be IN while-loop, not like keyDown
                            while not self.eog_reader.signal.empty():
                                try:
                                    self.eog_reader.signal.get_nowait()  #M: clear queue to avoid getting old signals during cooldown
                                except:
                                    break
                                
                            time.sleep(0.01) #M: to avoid getting cleared millions of times (CPU 1000) ## could cause lagging-problems if working w moveRel

                        pyautogui.moveRel(0, -self.speed)
                        while not self.eog_reader.signal.empty():
                            self.checking_opposite_direction, self.timestamp = self.eog_reader.signal.get()
                            if self.checking_opposite_direction == "up":
                                self.upPressed = False
                                self.up_start_time = 0
                                break

                            time.sleep(0.01)

            if self.direction == "down":
                if self.upPressed:
                    self.upPressed = False
                elif not self.downPressed:
                    self.downPressed = True
                    self.down_start_time = time.time()

                    while self.downPressed == True:
                        while time.time() - self.down_start_time < config.TURNING_COOLDOWN:  #M: COOLDOWN: Ignore opposite direction signal for 0.5 seconds
                            pyautogui.moveRel(0, self.speed) #M: moves 1 time by 10 pixels --> has to be IN while-loop, not like keyDown
                            while not self.eog_reader.signal.empty():
                                try:
                                    self.eog_reader.signal.get_nowait()  #M: clear queue to avoid getting old signals during cooldown
                                except:
                                    break
                                
                            time.sleep(0.01) #M: to avoid getting cleared millions of times (CPU 1000) ## could cause lagging-problems if working w moveRel

                        pyautogui.moveRel(0, self.speed)
                        while not self.eog_reader.signal.empty():
                            self.checking_opposite_direction, self.timestamp = self.eog_reader.signal.get()
                            if self.checking_opposite_direction == "down":
                                self.downPressed = False
                                self.down_start_time = 0
                                break

                            time.sleep(0.01)

if __name__ == "main":
    mouseControl = MouseControl()


                    






