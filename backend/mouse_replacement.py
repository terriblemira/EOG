#creates continuous movements in direction of eye movement signal until looking in another direction
#TODO: insert stop condition for continuous movement (left --> stop when right; right --> stop when left etc. BUT AFTER Cooldown (evtl. already eog_reader cooldown enough) )
import pyautogui
import eog_reader
import threading
import time

class MouseMover:
    def __init__(self, speed = 5):
        self.speed = speed
        self.running = False
        self.current_direction = None
        self.mouse_thread = None
    
        pyautogui.FAILSAFE = True     #M: stops when mouse moved to corner 

        #M: method run as thread
    def move_continuously(self):
        try:
             counter = 0
             while self.running:
                counter += 1

                #M: Get new direction signal every 0.1 s (10 iterations of 0.01s sleep)
                #TODO: make sure next signal in queue that gets is current one, not old one (clear queue before getting new one?)
                if counter % 10 == 0:
                    if not eog_reader.signal.empty():
                        self.direction = eog_reader.signal.get()
                        print(f"direction {self.direction}")
                
                #M: not indented as much bc: Move EVERY iteration, not just % 10 == 0 (every 10th)
                if self.direction == "left":
                    pyautogui.moveRel(-self.speed, 0)
                elif self.direction == "right":
                    pyautogui.moveRel(self.speed, 0)
                elif self.direction == "up":
                    pyautogui.moveRel(0, -self.speed)
                elif self.direction == "down":
                    pyautogui.moveRel(0, self.speed)

                time.sleep(0.01)  # Adjust the sleep time for smoother or faster movement

        except Exception as e:
            print(f"Error in move_continuously: {e}")  

        
    def initiate_move_continuously(self):
        self.running = True

        if not eog_reader.signal.empty():
            self.direction = eog_reader.signal.get()

        self.mouse_thread = threading.Thread(target=self.move_continuously, daemon=True) #M: run function move_cont. as thread (like while True: move_cont() but not blocking loop)
        self.mouse_thread.start()
        print(f"Started moving {self.direction}")

    def stop_move_continuously(self):
        if self.running:
            self.running = False
            if self.mouse_thread: #M: if thread is alive
                self.mouse_thread.join() #M: stops main thread (from "signaling mousethread to stop" until "mouse thread actually stops")
        print("Mouse stopped moving, main thread continuing now")

if __name__ == "__main__":
    mouseMover = MouseMover()
    #Start thread
    mouseMover.initiate_move_continuously()

#M Keep the main thread alive to allow continuous movement
    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        mouseMover.stop_move_continuously()
        print("Program terminated.")


