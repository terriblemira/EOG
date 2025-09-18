# EOG
Electrooculography (EOG) signals use BCI technology to detect eye movements. We are using it to setup a gaming platform where you can use eyes to control movements.

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
