import argparse
import asyncio
import logging
import os
import random
import socket
import time
from typing import Optional

import httpx
import orjson
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

API_BASE = os.getenv("TOGETHER_API_BASE_URL", "api.together.ai")
RETRIEVE_URL = f"https://{API_BASE}/internal/v1/rabbitmq/retrieve"
UPDATE_URL = f"https://{API_BASE}/internal/v1/videos/status"

HOSTNAME = socket.gethostname()
try:
    VERSION = open("VERSION").read().strip()
except FileNotFoundError:
    VERSION = ""


class OrjsonResponse(JSONResponse):
    def render(self, content: "Any") -> bytes:
        return orjson.dumps(content)


class Sprocket:
    def setup(self) -> None:
        raise NotImplementedError

    def predict(self, args: dict) -> dict:
        raise NotImplementedError


class Runner:
    def __init__(self, sprocket: Sprocket, model_name: str) -> None:
        api_key = os.getenv("TOGETHER_API_KEY")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        if VERSION:
            headers["x-worker-version"] = VERSION
        self.client = httpx.AsyncClient(headers=headers)
        self.shutdown_event = asyncio.Event()
        self.sprocket = sprocket
        self.model_name = model_name
        self.busy = False
        self.queue_mode = False

    async def get_job(self) -> dict:
        params = {"timeout": "5s", "model": self.model_name, "hostname": HOSTNAME}
        start_time = time.time()
        while not self.shutdown_event.is_set():
            try:
                logger.info("waiting for job from remote queue")
                headers = {"x-idle-time": str(time.time() - start_time)}
                response = await self.client.get(
                    RETRIEVE_URL, headers=headers, params=params
                )
                response.raise_for_status()
                resp = response.json()
                if resp.get("data"):
                    return resp["data"]
                logger.info("no job found in remote queue")
            except httpx.HTTPError as e:
                logger.error(f"failed to get job from remote queue: {e}")
            await asyncio.sleep(random.random() / 5)
        raise Exception("exit requested")

    async def update_job_status(
        self,
        request_id: str,
        status: str,
        outputs: Optional[dict] = None,
        info: Optional[dict] = None,
    ) -> bool:
        data = {
            "model": self.model_name,
            "request_id": request_id,
            "status": status,
        }
        if outputs:
            data["payload"] = outputs
        if info:
            data["info"] = info
        logger.info(f"updating remote job status: {data}")
        for i in range(3):
            try:
                response = await self.client.post(UPDATE_URL, json=data)
                if response.status_code == 200:
                    return True
                response.raise_for_status()
            except httpx.HTTPError as e:
                logger.error(f"failed to update job status: {e}")
        logger.error("failed to update job status after 3 retries")
        return False

    async def run_one_job(self) -> None:
        job = await self.get_job()
        request_id = job["request_id"]
        self.busy = True
        try:
            output = self.sprocket.predict(job["payload"])
        except Exception as e:
            logger.error(f"Job {request_id} failed")
            await self.update_job_status(request_id, "failed", info={"error": str(e)})
        else:
            logger.info(f"Job {request_id} finished")
            await self.update_job_status(request_id, "done", outputs=output)
        finally:
            self.busy = False

    async def run_queue_worker(self) -> None:
        try:
            while not self.shutdown_event.is_set():
                await self.run_one_job()
        except Exception as e:
            logger.error(f"Queue worker error: {e}")

    def create_app(self) -> Starlette:
        app = Starlette()

        @app.on_event("startup")
        async def startup():
            self.sprocket.setup()
            if self.queue_mode:
                asyncio.create_task(self.run_queue_worker())

        @app.on_event("shutdown")
        async def shutdown():
            self.shutdown_event.set()
            await self.client.aclose()

        @app.route("/health")
        async def health(request: Request) -> JSONResponse:
            return JSONResponse({"status": "healthy"})

        @app.route("/metrics")
        async def metrics(request: Request) -> PlainTextResponse:
            busy_value = 1.0 if self.busy else 0.0
            return PlainTextResponse(f"requests_inflight {busy_value}")

        @app.route("/generate", methods=["POST"])
        async def generate(request: Request) -> JSONResponse:
            if self.queue_mode:
                return JSONResponse(
                    {"error": "Worker is in queue mode"}, status_code=503
                )

            # TODO: Future support for async/batching/concurrency
            if self.busy:
                return JSONResponse({"error": "Worker is busy"}, status_code=503)

            self.busy = True
            try:
                data = await request.json()
                result = self.sprocket.predict(data)
                return OrjsonResponse(result)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
            finally:
                self.busy = False

        return app

    async def run(self) -> None:
        parser = argparse.ArgumentParser(description="Sprocket worker")
        parser.add_argument(
            "--queue",
            action="store_true",
            help="Enable queue mode (default: HTTP server)",
        )
        parser.add_argument(
            "--model", default="default", help="Model name for queue mode"
        )
        parser.add_argument(
            "--port", type=int, default=8000, help="Port for HTTP server mode"
        )
        args = parser.parse_args()

        self.queue_mode = args.queue

        if args.queue:
            logger.info("Starting in queue mode with HTTP server")
        else:
            logger.info("Starting HTTP server only")

        app = self.create_app()
        config = uvicorn.Config(
            app=app, host="0.0.0.0", port=args.port, log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
