# EOG
Electrooculography (EOG) signals use BCI technology to detect eye movements. We are using it to setup a gaming platform where you can use eyes to control movements.

## Game Glasses

Bare-bones web app skeleton for controlling simple games via EOG (future).
Tech: FastAPI + Jinja templates + static JS/CSS. WebSocket placeholder for LSL.

## Quick Start

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
