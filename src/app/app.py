import os
import sys
import socket
import threading
import webbrowser

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.inference import predict


app = FastAPI(title="Lab Delay Prediction API")


@app.get("/")
def root():
    return {"status": "ok", "service": "lab-delay-prediction"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ui")
def serve_ui():
    ui_path = os.path.join(os.path.dirname(__file__), "ui.html")
    return FileResponse(ui_path)


class LabData(BaseModel):
    labname: str
    gender: str
    age: str
    unittype: str
    recent_diagnosis: str | None = None
    result_year: int = 2026
    result_month: int = 1
    result_day: int = 1
    result_hour: int = 0
    result_weekday: int = 0
    admissionweight: float | None = None
    lab_workload_last_hour: int


@app.post("/predict")
def api_predict(data: LabData):
    return predict(data.model_dump())


def _find_available_port(port: int) -> int:
    for candidate in range(port, port + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", candidate)) != 0:
                return candidate
    return port


if __name__ == "__main__":
    host = os.environ.get("APP_HOST", "0.0.0.0")
    start_port = int(os.environ.get("PORT", "8000"))
    port = _find_available_port(start_port)

    print(f"Starting server on http://127.0.0.1:{port}/ui")

    if os.environ.get("OPEN_BROWSER", "1") == "1":
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}/ui")).start()

    uvicorn.run(app, host=host, port=port)