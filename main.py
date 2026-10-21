"""Application entry point.

Run locally (VS Code / terminal):
    uvicorn main:app --reload --port 8000
or:
    python main.py
"""
from config.settings import Settings
from api.app import create_app

app = create_app(Settings.load())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
