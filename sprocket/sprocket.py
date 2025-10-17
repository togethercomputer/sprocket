import argparse
import asyncio
import logging
import multiprocessing.connection
import os
import pathlib
import random
import socket
import subprocess
import sys
import time
import traceback
from collections import namedtuple
from collections.abc import Awaitable
from typing import Any, Optional

import httpx
import orjson
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from .async_connections import ConnectionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

API_BASE = os.getenv("TOGETHER_API_BASE_URL", "api.together.ai")
RETRIEVE_URL = f"https://{API_BASE}/internal/v1/rabbitmq/retrieve"
UPDATE_URL = f"https://{API_BASE}/internal/v1/videos/status"
UPLOAD_URL = f"https://{API_BASE}/v1/storage/upload-request"

HOSTNAME = socket.gethostname()
try:
    VERSION = open("VERSION").read().strip()
except FileNotFoundError:
    VERSION = ""

SHUTDOWN_REQUESTED = pathlib.Path(".shutdown_requested")
SHUTDOWN_NOW = pathlib.Path(".shutdown_now")


class MetricsEndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/metrics") == -1


logging.getLogger("uvicorn.access").addFilter(MetricsEndpointFilter())


class FileOutput(pathlib.PosixPath):
    "Output file to be uploaded"


class OrjsonResponse(JSONResponse):
    def render(self, content: "Any") -> bytes:
        return orjson.dumps(content)


class Sprocket:
    def setup(self) -> None:
        raise NotImplementedError

    def predict(self, args: dict) -> dict:
        raise NotImplementedError

    def shutdown(self) -> None | Awaitable[None]:
        pass


class AsyncSprocket:
    async def setup(self) -> None:
        raise NotImplementedError

    async def predict(self, args: dict) -> dict:
        raise NotImplementedError


class Runner:
    def __init__(self, sprocket: Sprocket | AsyncSprocket, model_name: str) -> None:
        api_key = os.getenv("TOGETHER_API_KEY")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        if VERSION:
            headers["x-worker-version"] = VERSION
        self.client = httpx.AsyncClient(headers=headers, timeout=None)
        self.sprocket = sprocket
        self.model_name = model_name
        self.busy = False
        self.queue_mode = False
        self.healthy = False

    async def get_job(self) -> dict:
        params = {"timeout": "5s", "model": self.model_name, "hostname": HOSTNAME}
        start_time = time.time()
        while not SHUTDOWN_REQUESTED.exists():
            try:
                logger.info(f"waiting for job from remote queue {self.model_name}")
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
                logger.error(f"failed to get job from remote queue: {repr(e)}")
            await asyncio.sleep(random.random() / 5)
        raise Exception("exit requested")

    async def upload_file(self, request_id: str, path: FileOutput) -> str:
        try:
            resp = await self.client.post(UPLOAD_URL, json={"filename": request_id + "-" + path.name})
            await self.client.put(resp.json()["upload_url"]["url"], content=path.open("rb").read())
        except:
            traceback.print_exc()
            raise
        return f"{API_BASE}/v1/storage/{request_id}-{path.name}"

    async def handle_output_upload(self, request_id: str, output: dict) -> dict:
        return {
            k: await self.upload_file(request_id, v) if isinstance(v, FileOutput) else v
            for k, v in output.items()
        }

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
            "payload": outputs or {},
        }
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
            if isinstance(self.sprocket, AsyncSprocket):
                output = await self.sprocket.predict(job["payload"])
            else:
                output = self.sprocket.predict(job["payload"])
        except Exception as e:
            logger.error(f"Job {request_id} failed")
            await self.update_job_status(request_id, "failed", info={"error": repr(e)})
        else:
            logger.info(f"Job {request_id} finished")
            output = await self.handle_output_upload(request_id, output)
            await self.update_job_status(request_id, "done", outputs=output)
        finally:
            self.busy = False

    async def run_queue_worker(self) -> None:
        try:
            while not SHUTDOWN_REQUESTED.exists():
                await self.run_one_job()
            # the current job is complete, it's safe to exit without reading any more events
            if SHUTDOWN_REQUESTED.exists():
                SHUTDOWN_NOW.touch()
        except Exception as e:
            logger.error(f"Queue worker error: {e}")

    def create_app(self) -> Starlette:
        app = Starlette()

        @app.on_event("startup")
        async def startup() -> None:
            if isinstance(self.sprocket, AsyncSprocket):
                await self.sprocket.setup()
            else:
                self.sprocket.setup()
            if self.queue_mode:
                asyncio.create_task(self.run_queue_worker())
            self.healthy = True

        @app.on_event("shutdown")
        async def shutdown() -> None:
            self.healthy = False
            SHUTDOWN_REQUESTED.touch()
            await self.client.aclose()

        @app.route("/health")
        async def health(request: Request) -> JSONResponse:
            if self.healthy:
                return JSONResponse({"status": "healthy"})
            else:
                return JSONResponse({"status": "unhealthy"}, status_code=503)

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
                if isinstance(self.sprocket, AsyncSprocket):
                    result = await self.sprocket.predict(data)
                else:
                    result = self.sprocket.predict(data)
                fake_request_id = uuid.uuid4()
                result = await self.handle_output_upload(fake_request_id, result)
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


# we want something like that that we can import from random modeling code without having to pass anything
# even better if it's a file with atomic appends or something
# @(lambda x: x())
# class emit_anywhere:
#     def __call__(self, msg: "Any") -> None:
#         self.conn.send(msg)
#     def set(self, conn: Connection) -> None:
#         self.conn = conn

ChildToWorkerMessage = namedtuple("ChildToWorkerMessage", ["rank", "type", "arg"])


class ChildRunner:
    def __init__(self, sprocket: Sprocket, model_name: str) -> None:
        self.sprocket = sprocket
        self.model_name = model_name
        self.busy = False

    def run(self) -> None:
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Child runner started for local rank {local_rank}")
        try:
            self.conn = multiprocessing.connection.Client(os.environ["SPROCKET_SOCKET"])
            self.conn.send(ChildToWorkerMessage(local_rank, "setup_start", None))
            try:
                self.sprocket.setup()
            except Exception as e:
                self.conn.send(ChildToWorkerMessage(local_rank, "setup_error", e))
                return
            self.conn.send(ChildToWorkerMessage(local_rank, "setup_done", None))
            while not SHUTDOWN_REQUESTED.exists():
                try:
                    args = self.conn.recv()
                except EOFError:
                    # graceful exit
                    return
                try:
                    output = self.sprocket.predict(args)
                except Exception as e:
                    traceback.print_exc()
                    # I sure hope that's picklable
                    self.conn.send(ChildToWorkerMessage(local_rank, "error", e))
                else:
                    self.conn.send(ChildToWorkerMessage(local_rank, "output", output))
        except Exception as e:
            logger.error(f"Runner error: {e}")


# fixme: rearrange runner so it's easier to extend like this instead of needing nested sprockets
class TorchRunSprocket(AsyncSprocket):
    async def setup(self) -> None:
        sprocket_socket = f"/tmp/sprocket-{os.getpid()}"
        num_processes = int(os.getenv("WORLD_SIZE", 1))
        torchrun_args = [
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={num_processes}",
        ]
        # run torchrun
        os.environ["SPROCKET_SOCKET"] = sprocket_socket

        self.connection_manager = ConnectionManager[ChildToWorkerMessage](shutdown_path=SHUTDOWN_NOW)
        await self.connection_manager.run(sprocket_socket)

        print("Starting worker processes")
        self.torchrun_proc = await asyncio.create_subprocess_exec(
            "torchrun",
            *torchrun_args,
            *sys.argv,
            stdin=subprocess.DEVNULL,
            stdout=sys.stdout,
            stderr=sys.stderr,
            close_fds=True,
        )
        print("Awaiting connections from workers")
        await self.connection_manager.wait_for_connections(num_processes)
        setup_starts = 0
        setup_dones = 0
        print("Awaiting worker setup")
        async for msg in self.connection_manager.gather():
            if msg.type == "setup_start":
                setup_starts += 1
            elif msg.type == "setup_done":
                setup_dones += 1
            elif msg.type == "setup_error":
                raise msg.arg
            else:
                raise ValueError(f"invalid msg {msg}")
            if setup_starts == setup_dones == num_processes:
                break
        print("Worker setup complete")

    async def predict(self, args: dict) -> dict:
        await self.connection_manager.broadcast(args)
        async for msg in self.connection_manager.gather():
            if not msg.arg:
                continue  # ignore ranks returning None
            if msg.type == "error":
                raise msg.arg
            if msg.type == "output":
                # FIXME: get ready event from every event before submitting new jobs so we catch errors + outputs
                return msg.arg
        raise Exception("shutting down")
        # also handle upload
        # self.update_job_status(msg)

    async def shutdown(self) -> None:
        # todo: timeout
        if self.torchrun_proc:
            self.torchrun_proc.terminate()
            await self.torchrun_proc.wait()


def run(sprocket: Sprocket, name: str, use_torchrun: bool = False) -> None:
    SHUTDOWN_NOW.unlink(missing_ok=True)
    SHUTDOWN_REQUESTED.unlink(missing_ok=True)
    if "LOCAL_RANK" in os.environ:
        if "SPROCKET_SOCKET" not in os.environ:
            logger.error("Please don't start sprocket processes in torchrun directly")
            sys.exit(1)
        ChildRunner(sprocket, name).run()
    elif use_torchrun:
        asyncio.run(Runner(TorchRunSprocket(), name).run())
    else:
        asyncio.run(Runner(sprocket, name).run())
