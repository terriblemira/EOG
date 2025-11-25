# EOG
Electrooculography (EOG) signals use BCI technology to detect eye movements. We are using it to setup a gaming platform where you can use eyes to control movements.

MINECRAFT CONNECTION IDEAS:
1. General:
- 1st person
- important moves: lefty/righty/jumpy/sneeky/hit(y)/klick(y)/dropn/open inventoryy/runn
- moving eyes back to center can't be detected --> COOLDOWN!!
- possible problem: difference between UP and WINK: Tresholds may be similar/same & using pattern detection (2 opposite peaks in UP may result in at least 0.5 s delay)
- during calibration: player needs to move nearer to screen in order for small "lookarounds" not being detected as movements (only "extremer" ones)
2. Key Ideas:
- UP: Jump
- DOWN: Sneek
- LEFT/RIGHT: turns left/right until you look back to center (detected as opposite direction)
- DOUBLE BLINK: Walk (DOUBLE BLINK AGAIN: Stop)
- WINK: Click
- LONG BLINK: open inventory --> LEFT/RIGHT & back to center: no turning but selection changes to left/right continously until looking opposite (not too fast) --> after clicking: inventory automatically closes = back to turning
- WINK FAST (more than once): hitting


## Game Glasses

Web-app to host games and more to be controlled by EOG

## Quick Start

[Insert instructions on connecting mentalab and running the LSL stream]

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app:app
