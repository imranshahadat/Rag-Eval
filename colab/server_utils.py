"""Helpers for running the API server inside Google Colab."""
from __future__ import annotations

import threading
import time

import requests
import uvicorn
from pyngrok import ngrok


class ServerManager:
    """Runs the FastAPI app in a background daemon thread; supports graceful stop
    and an optional ngrok public tunnel."""

    def __init__(self, port: int = 8000):
        self.port = port
        self.health_url = f"http://127.0.0.1:{port}/health"
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def start(self, app, timeout: int = 90) -> bool:
        """Start the server and wait until the health endpoint responds."""
        if self._thread and self._thread.is_alive():
            print(f"Server already running on port {self.port}.")
            return True

        # uvicorn.Server (not uvicorn.run) so we can stop it gracefully later
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="info")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        print("Waiting for server startup ...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if requests.get(self.health_url, timeout=2).ok:
                    print(f"API is ready at http://127.0.0.1:{self.port}")
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)

        print("Server did not become healthy in time. Check the logs above.")
        return False

    def stop(self) -> None:
        """Graceful shutdown so lifespan cleanup (db.close) actually runs."""
        if self._server is None or self._thread is None or not self._thread.is_alive():
            print("No server is running.")
            return
        self._server.should_exit = True
        self._thread.join(timeout=15)
        print("Server stopped.")

    def connect_ngrok(self, auth_token: str | None = None) -> str:
        """Open an ngrok tunnel to the local server and print the public URL."""
        if auth_token:
            ngrok.set_auth_token(auth_token)

        # Close stale tunnels so repeated runs do not accumulate dead links
        for tunnel in ngrok.get_tunnels():
            ngrok.disconnect(tunnel.public_url)

        public_url = ngrok.connect(self.port).public_url
        print(f"Public URL: {public_url}")
        print(f"Swagger UI: {public_url}/docs")
        return public_url
