import argparse
import asyncio
import logging
import multiprocessing.connection
import os
import pathlib
import pickle
import random
import socket
import struct
import subprocess
import sys
import time
import traceback
import uuid
from collections import namedtuple
from collections.abc import Awaitable
from typing import Any, AsyncIterator, Optional, Type
from urllib.parse import urlparse

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


class QueueClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        api_key = os.getenv("TOGETHER_API_KEY")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        if VERSION:
            headers["x-worker-version"] = VERSION
        self.client = httpx.AsyncClient(headers=headers, timeout=None)

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

    async def upload_file(self, request_id: str, path: FileOutput) -> str:
        try:
            resp = await self.client.post(
                UPLOAD_URL, json={"filename": request_id + "-" + path.name}
            )
            await self.client.put(
                resp.json()["upload_url"]["url"], content=path.open("rb").read()
            )
        except:
            traceback.print_exc()
            raise
        return f"{API_BASE}/v1/storage/{request_id}-{path.name}"


class OrjsonResponse(JSONResponse):
    def render(self, content: "Any") -> bytes:
        return orjson.dumps(content)


class InputOutputProcessor:
    def process_input_file(self, resp: httpx.Response, dst: pathlib.Path) -> None:
        """
        overwrite this to add processing after files are downloaded
        """
        open(dst, "wb").write(resp.content)

    async def finalize(self, request_id: str, inputs: dict, outputs: dict) -> dict:
        """
        this function may be called concurrently with starting the next job
        in torchrun mode, this is called in the parent process
        """
        return outputs


class Sprocket:
    processor: Type[InputOutputProcessor] = InputOutputProcessor

    def setup(self) -> None:
        raise NotImplementedError

    def predict(self, args: dict) -> dict:
        raise NotImplementedError

    def shutdown(self) -> None | Awaitable[None]:
        pass


class AsyncSprocket:
    processor: Type[InputOutputProcessor] = InputOutputProcessor

    async def setup(self) -> None:
        raise NotImplementedError

    async def predict(self, args: dict) -> dict:
        raise NotImplementedError


class Runner:
    def __init__(self, sprocket: Sprocket | AsyncSprocket, model_name: str) -> None:
        self.queue_client = QueueClient(model_name)
        # borrow httpx client
        self.download_client = httpx.AsyncClient(timeout=None, follow_redirects=True)
        self.sprocket = sprocket
        # create InputOutputProcessor, potentially overriden by sprocket
        self.io_processor = sprocket.processor()
        self.busy = False
        self.queue_mode = False
        self.healthy = False

    async def download_file(self, url: str) -> pathlib.Path:
        # replace with more sophisticated download later
        dst = pathlib.Path("inputs/" + os.path.basename(urlparse(url).path))
        resp = self.download_client.get(url)
        resp.raise_for_status()
        self.io_processor.process_input_file(resp, dst)
        return dst

    async def handle_job(self, inputs: dict, request_id: str) -> dict:
        logger.info(f"queue inputs: {inputs}")
        job_start_time = time.time()
        self.busy = True
        downloaded_paths = []
        try:
            # change urls in values to paths
            for k, v in inputs.items():
                if isinstance(v, str) and v.startswith("https://"):
                    downloaded_paths.append(await self.download_file(v))
                    inputs[k] = str(downloaded_paths[-1])
            if isinstance(self.sprocket, AsyncSprocket):
                output = await self.sprocket.predict(inputs)
            else:
                output = self.sprocket.predict(inputs)
            logger.info("total predict run time: {time.time() - job_start_time}")
            output = await self.io_processor.finalize(request_id, inputs, output)
            return {
                k: await self.upload_file(request_id, v)
                if isinstance(v, FileOutput)
                else v
                for k, v in output.items()
            }
        finally:
            self.busy = False
            for path in downloaded_paths:
                path.unlink(missing_ok=True)

    async def run_one_job(self) -> None:
        job = await self.queue_client.get_job()
        request_id = job["request_id"]
        try:
            output = await self.handle_job(job["payload"], request_id)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Job {request_id} failed")
            await self.queue_client.update_job_status(
                request_id, "failed", info={"error": repr(e)}
            )
        else:
            logger.info(f"Job {request_id} finished")
            await self.queue_client.update_job_status(
                request_id, "done", outputs=output
            )

    async def run_queue_worker(self) -> None:
        try:
            while not SHUTDOWN_REQUESTED.exists():
                await self.run_one_job()
            # the current job is complete, it's safe to exit without reading any more events
            if SHUTDOWN_REQUESTED.exists():
                SHUTDOWN_NOW.touch()
        except Exception as e:
            logger.error(f"Queue worker error: {repr(e)}")

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
            # TODO: Future support for async/batching/concurrency
            if self.busy:
                return JSONResponse({"error": "Worker is busy"}, status_code=503)

            try:
                data = await request.json()
                # change to uuidv7 in python3.14
                result = await self.handle_job(data, request_id=str(uuid.uuid4()))
                return OrjsonResponse(result)
            except Exception as e:
                traceback.print_exc()
                return JSONResponse({"error": str(e)}, status_code=500)

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
                finally:
                    self.conn.send(
                        ChildToWorkerMessage(local_rank, "predict_done", None)
                    )
        except Exception as e:
            logger.error(f"Runner error: {repr(e)}")


class AsyncConnection:
    def __init__(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.reader = reader
        self.writer = writer

    async def recv(self) -> ChildToWorkerMessage:
        # this might need to be in an asyncio.gather or have a timeout to exit cleanly
        data = await self.reader.readexactly(4)
        (size,) = struct.unpack("!i", data)
        if size == -1:
            # watch out! size > 0x7FFFFFFF
            data = await self.reader.readexactly(8)
            (size,) = struct.unpack("!Q", data)
        data = await self.reader.readexactly(size)
        return pickle.loads(data)

    async def send(self, message: ChildToWorkerMessage) -> None:
        data = pickle.dumps(message)
        size = len(data)
        if size < 2**31:
            header = struct.pack("!i", size)
        else:
            header = struct.pack("!iQ", -1, size)
        self.writer.write(header)
        self.writer.write(data)
        await self.writer.drain()


class ConnectionManager:
    def __init__(self, num_processes: int) -> None:
        self.conns: list[AsyncConnection] = []
        self.connected = asyncio.Condition()
        self.queue: asyncio.Queue[ChildToWorkerMessage] = asyncio.Queue()
        self.num_processes = num_processes

    async def run(self, sprocket_socket: str) -> None:
        async def start_connection(
            r: asyncio.StreamReader, w: asyncio.StreamWriter
        ) -> None:
            c = AsyncConnection(r, w)
            self.conns.append(c)
            async with self.connected:
                self.connected.notify_all()
            while not SHUTDOWN_NOW.exists():
                await self.queue.put(await c.recv())

        self.listener = await asyncio.start_unix_server(
            start_connection, path=sprocket_socket
        )

    async def wait_for_connections(self) -> None:
        async with self.connected:
            await self.connected.wait_for(lambda: len(self.conns) == self.num_processes)

    async def broadcast(self, msg: Any) -> None:
        for conn in self.conns:
            await conn.send(msg)

    async def gather(self) -> "AsyncIterator[ChildToWorkerMessage]":
        while not SHUTDOWN_NOW.exists():
            yield await self.queue.get()


# fixme: rearrange runner so it's easier to extend like this instead of needing nested sprockets
class TorchRunSprocket(AsyncSprocket):
    async def setup(self) -> None:
        sprocket_socket = f"/tmp/sprocket-{os.getpid()}"
        self.num_processes = int(os.getenv("WORLD_SIZE", 1))
        torchrun_args = [
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={self.num_processes}",
        ]
        # run torchrun
        os.environ["SPROCKET_SOCKET"] = sprocket_socket

        self.connection_manager = ConnectionManager(self.num_processes)
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
        await self.connection_manager.wait_for_connections()
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
            if setup_starts == setup_dones == self.num_processes:
                break
        print("Worker setup complete")
        # this can be useful for profiling with nsys
        iter_count = int(os.getenv("RUN_FOR_PROFILING", "0"))
        if iter_count:
            await self.predict({})
            times = []
            for i in range(iter_count):
                st = time.time()
                await self.predict({})
                times.append(time.time() - st)
            print(f"===predict times===: {times}")

            SHUTDOWN_REQUESTED.touch()
            sys.exit(0)
        else:
            print("not profiling")

    async def predict(self, args: dict) -> dict:
        await self.connection_manager.broadcast(args)
        predict_dones = 0
        result = None
        err = None
        async for msg in self.connection_manager.gather():
            if msg.type == "error":
                err = msg.arg  # we will raise the last error we receive
            if msg.type == "output":
                if msg.arg:  # ignore ranks returning None
                    result = msg.arg
            elif msg.type == "predict_done":
                predict_dones += 1
            if predict_dones == self.num_processes:
                if err:
                    raise err
                if result:
                    return result
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
        # copy over the processor to the sprocket that will actually be used
        TorchRunSprocket.processor = sprocket.processor
        asyncio.run(Runner(TorchRunSprocket(), name).run())
    else:
        asyncio.run(Runner(sprocket, name).run())
