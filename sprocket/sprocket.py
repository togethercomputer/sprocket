import argparse
import asyncio
import contextlib
import dataclasses
import enum
import importlib.metadata
import logging
import multiprocessing.connection
import os
import pickle
import random
import signal
import socket
import struct
import subprocess
import sys
import time
import traceback
import uuid
from asyncio import StreamReader, StreamWriter
from pathlib import Path, PosixPath
from typing import Any, AsyncIterator, Optional, Type
from urllib.parse import urlparse

import httpx
import orjson
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route
from starlette.responses import JSONResponse, PlainTextResponse

fmt = "{levelname} {module}:{lineno}: {message}"
logging.basicConfig(level=logging.INFO, format=fmt, style="{")
logger = logging.getLogger()

API_BASE = os.getenv("TOGETHER_API_BASE_URL", "https://api.together.ai")
if not API_BASE.startswith("https://"):
    API_BASE = f"https://{API_BASE}"
RETRIEVE_URL = f"{API_BASE}/internal/v1/queue/retrieve"
UPDATE_URL = f"{API_BASE}/internal/v1/videos/status"
UPLOAD_URL = f"{API_BASE}/v1/storage/upload-request"

# make sure we can complete or time out jobs before we get killed
MAX_ASYNC_PREDICT_TIME = int(os.getenv("TERMINATION_GRACE_PERIOD_SECONDS", "300")) - 1

HOSTNAME = socket.gethostname()
try:
    SPROCKET_VERSION = f" {__package__}/{importlib.metadata.version(__package__ or '')}"
except (ValueError, importlib.metadata.PackageNotFoundError):
    SPROCKET_VERSION = ""

try:
    VERSION = open("VERSION").read().strip()
except FileNotFoundError:
    VERSION = ""

SHUTDOWN_REQUESTED = Path(".shutdown_requested")
SHUTDOWN_NOW = Path(".shutdown_now")
FATAL_ERROR = Path(".fatal_error")


class NoMetrics(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/metrics") == -1


logging.getLogger("uvicorn.access").addFilter(NoMetrics())


class FileOutput(PosixPath):
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
        agent_version_string = f"{SPROCKET_VERSION} {model_name}/{VERSION or 'unknown'}"
        self.client.headers["User-Agent"] += agent_version_string

    async def get_job(self, timeout: Optional[int] = None) -> dict:
        params = {"timeout": "5s", "model": self.model_name, "hostname": HOSTNAME}
        if timeout:
            params["claim_timeout"] = timeout
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
            req = {"filename": request_id + "-" + path.name}
            resp = await self.client.post(UPLOAD_URL, json=req)
            url = resp.json()["upload_url"]["url"]
            await self.client.put(url, content=path.open("rb").read())
        except Exception:
            traceback.print_exc()
            raise
        return f"{API_BASE}/v1/storage/{request_id}-{path.name}"


class OrjsonResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return orjson.dumps(content)


class InputOutputProcessor:
    def process_input_file(self, response: httpx.Response, dest_path: Path) -> None:
        """
        overwrite this to add processing after files are downloaded
        """
        open(dest_path, "wb").write(response.content)

    async def finalize(self, request_id: str, inputs: dict, outputs: dict) -> dict:
        """
        this function may be called concurrently with starting the next job
        in torchrun mode, this is called in the parent process
        """
        return outputs


class Sprocket:
    processor: Type[InputOutputProcessor] = InputOutputProcessor
    warmup_inputs: list[dict] = []  # inputs to run during warmup/cache generation

    def setup(self) -> None:
        raise NotImplementedError

    def predict(self, args: dict) -> dict:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass


class AsyncSprocket:
    processor: Type[InputOutputProcessor] = InputOutputProcessor
    warmup_inputs: list[dict] = []

    async def setup(self) -> None:
        raise NotImplementedError

    async def predict(self, args: dict) -> dict:
        raise NotImplementedError

    async def shutdown(self) -> None:
        pass


class Runner:
    def __init__(self, sprocket: "Sprocket | AsyncSprocket", model_name: str) -> None:
        self.queue_client = QueueClient(model_name)
        self.download_client = httpx.AsyncClient(timeout=None, follow_redirects=True)
        self.sprocket = sprocket
        # create InputOutputProcessor, potentially overriden by sprocket
        self.io_processor = sprocket.processor()
        self.busy = False
        self.queue_mode = False
        self.healthy = False

    async def download_file(self, url: str) -> Path:
        dst = Path("inputs/" + os.path.basename(urlparse(url).path))
        resp = await self.download_client.get(url)
        resp.raise_for_status()
        self.io_processor.process_input_file(resp, dst)
        return dst

    async def run_predict_with_timeout(self, inputs: dict) -> dict:
        if isinstance(self.sprocket, Sprocket):
            return self.sprocket.predict(inputs)
        try:
            predict_future = self.sprocket.predict(inputs)
            return await asyncio.wait_for(predict_future, MAX_ASYNC_PREDICT_TIME)
        except TimeoutError:
            # time's up, but the subprocess is still busy and we can't reset it
            # so we need to bail and restart cleanly
            FATAL_ERROR.touch()
            raise TimeoutError(f"Prediction took longer than {MAX_ASYNC_PREDICT_TIME}s")

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
            output = await self.run_predict_with_timeout(inputs)
            logger.info(f"total predict run time: {time.time() - job_start_time}")
            output = await self.io_processor.finalize(request_id, inputs, output)
            return {
                k: await self.queue_client.upload_file(request_id, v)
                if isinstance(v, FileOutput)
                else v
                for k, v in output.items()
            }
        finally:
            self.busy = False
            for path in downloaded_paths:
                path.unlink(missing_ok=True)

    async def refresh_job_claim(self, request_id: str) -> None:
        # don't try to update in the background if we're synchronous
        if not isinstance(self.sprocket, AsyncSprocket):
            return
        try:
            while True:
                await asyncio.sleep(45)
                if self.healthy:
                    await self.queue_client.update_job_status(request_id, "running")
        except asyncio.CancelledError:
            # task was cancelled, which is expected when job completes
            pass

    async def run_one_job(self) -> None:
        # short timeout if we can refresh in the background
        timeout = 90 if isinstance(self.sprocket, AsyncSprocket) else None
        job = await self.queue_client.get_job(timeout)
        request_id = job["request_id"]
        refresh_claim_task = asyncio.create_task(self.refresh_job_claim(request_id))
        try:
            try:
                output = await self.handle_job(job["payload"], request_id)
            finally:
                # we need to stop marking is as "running" before we set a terminal status
                refresh_claim_task.cancel()
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Job {request_id} failed")
            await self.queue_client.update_job_status(
                request_id, "failed", info={"error": repr(e)}
            )
            if isinstance(e, ConnectionResetError):
                logger.error("Connection lost, so probably children died")
                os.kill(os.getpid(), signal.SIGTERM)
            elif FATAL_ERROR.exists():
                logger.error("FATAL_ERROR is set")
                os.kill(os.getpid(), signal.SIGTERM)
        else:
            logger.info(f"Job {request_id} finished")
            await self.queue_client.update_job_status(
                request_id, "done", outputs=output
            )

    async def run_queue_worker(self) -> None:
        try:
            while not SHUTDOWN_REQUESTED.exists():
                await self.run_one_job()
        except Exception as e:
            logger.error(f"Queue worker error: {repr(e)}")
        finally:
            # the current job is complete, it's safe to exit without reading any more events
            if SHUTDOWN_REQUESTED.exists():
                logging.info("Current job ended while shutdown requested, ok to exit")
                SHUTDOWN_NOW.touch()

    async def handle_shutdown(self) -> None:
        "when we receive SIGTERM, this fn should block until we're ready to exit"
        logger.error("SIGTERM received, shutting down")
        self.healthy = False
        SHUTDOWN_REQUESTED.touch()

        # If idle, no need to wait for work completion
        if not self.busy:
            logger.info("Shutdown while idle, no need to wait")
            SHUTDOWN_NOW.touch()
        else:
            while not SHUTDOWN_NOW.exists():
                await asyncio.sleep(0.2)
            logger.info("Prediction complete after shutdown requested, exiting")

        logging.info("shutdown_now is finally set")

        # cleanup sprocket (terminates torchrun subprocesses if any)
        result = self.sprocket.shutdown()
        if result is not None:
            await result
        logger.info("Sprocket shutdown done")
        SHUTDOWN_NOW.touch()

    async def maybe_run_warmup(self) -> None:
        # RUN_AND_EXIT: run warmup inputs the specified number of times then exit
        # useful for building caches or profiling with nsys
        if run_and_exit := int(os.getenv("RUN_AND_EXIT", "0")):
            # warmup inputs default to {}
            warmup_inputs = (self.sprocket.warmup_inputs or [{}]) * run_and_exit
        else:
            # only if warmup inputs are set, run them once
            warmup_inputs = self.sprocket.warmup_inputs or []

        cold_start_time = time.time() - PROCESS_START_TIME
        times = []
        for input in warmup_inputs:
            start = time.time()
            if isinstance(self.sprocket, AsyncSprocket):
                await self.sprocket.predict(input)
            else:
                self.sprocket.predict(input)
            times.append(time.time() - start)
        print(f"===cold start time===: {cold_start_time:.2f}s")
        if times:
            print(f"===predict times===: {times}")
        if run_and_exit:
            SHUTDOWN_REQUESTED.touch()
            sys.exit(0)

    @contextlib.asynccontextmanager
    async def lifespan(self, _: Starlette) -> AsyncIterator[None]:
        if isinstance(self.sprocket, AsyncSprocket):
            await self.sprocket.setup()
        else:
            self.sprocket.setup()
        await self.maybe_run_warmup()
        if self.queue_mode:
            asyncio.create_task(self.run_queue_worker())
        self.healthy = True
        try:
            yield
        finally:
            await self.handle_shutdown()

    async def health_route(self, _: Request) -> JSONResponse:
        if isinstance(self.sprocket, TorchRunSprocket) and self.sprocket.torchrun_proc:
            if (retcode := self.sprocket.torchrun_proc.returncode) is not None:
                # this should be very unlikely - ConnectionManager should catch errors and exit before this happens
                logger.error(f"Torchrun process exited with retcode {retcode}!!")
                self.healthy = False
                FATAL_ERROR.touch()
        if self.healthy:
            return JSONResponse({"status": "healthy"})
        return JSONResponse({"status": "unhealthy"}, status_code=503)

    async def metrics_route(self, _: Request) -> PlainTextResponse:
        busy_value = 1.0 if self.busy else 0.0
        return PlainTextResponse(f"requests_inflight {busy_value}")

    async def generate_route(self, request: Request) -> JSONResponse:
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
        finally:
            if FATAL_ERROR.exists():
                logger.error("FATAL_ERROR is set")
                os.kill(os.getpid(), signal.SIGTERM)
            if SHUTDOWN_REQUESTED.exists():
                SHUTDOWN_NOW.touch()

    async def run(self) -> None:
        parser = argparse.ArgumentParser(description="Sprocket worker")
        parser.add_argument(
            "--queue",
            action="store_true",
            help="Enable queue mode (default: HTTP server)",
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

        routes = [
            Route("/generate", self.generate_route, methods=["POST"]),
            Route("/health", self.health_route),
            Route("/metrics", self.metrics_route),
        ]
        app = Starlette(routes=routes, lifespan=self.lifespan)
        config = uvicorn.Config(
            app=app, host="0.0.0.0", port=args.port, log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


# == mini IPC ==


class Message(str, enum.Enum):
    SETUP_START = "setup_start"
    SETUP_ERROR = "setup_error"
    SETUP_DONE = "setup_done"
    ERROR = "error"
    OUTPUT = "output"
    PREDICT_DONE = "predict_done"


@dataclasses.dataclass
class ChildMsg:
    rank: int
    type: Message
    arg: Any


@dataclasses.dataclass
class ChildRunner:
    sprocket: Sprocket
    model_name: str
    rank: int

    def run(self) -> None:
        try:
            conn = multiprocessing.connection.Client(os.environ["SPROCKET_SOCKET"])
        except KeyError:
            print("Please don't start sprocket processes in torchrun directly")
            sys.exit(1)
        except Exception as e:
            print(f"Child runner {self.rank} couldn't connect to parent: {repr(e)}")
            raise

        def send(m: Message, arg: Any = None) -> None:
            conn.send(ChildMsg(self.rank, m, arg))

        print(f"Child runner {self.rank} started")
        try:
            send(Message.SETUP_START)
            try:
                self.sprocket.setup()
            except Exception as e:
                send(Message.SETUP_ERROR, e)
                return
            send(Message.SETUP_DONE)
            while not SHUTDOWN_REQUESTED.exists():
                try:
                    args = conn.recv()
                except EOFError:  # graceful exit
                    print(f"Parent closed connection, rank {self.rank} exiting")
                    return
                try:
                    output = self.sprocket.predict(args)
                except Exception as e:
                    traceback.print_exc()
                    send(Message.ERROR, e)  # hope that error is picklable
                else:
                    send(Message.OUTPUT, output)
                finally:
                    send(Message.PREDICT_DONE)
                    print(f"Rank {self.rank}: PREDICT_DONE sent", flush=True)
        except Exception as e:
            print(f"Child runner {self.rank} error: {repr(e)}")
        state = f"shutdown requested: {SHUTDOWN_REQUESTED.exists()}, shutdown now: {SHUTDOWN_NOW.exists()}"
        print(f"Child runner {self.rank} exited. {state}")


class AsyncConnection:
    def __init__(self, r: StreamReader, w: StreamWriter) -> None:
        self.reader = r
        self.writer = w

    async def recv(self) -> ChildMsg:
        data = await self.reader.readexactly(4)
        (size,) = struct.unpack("!i", data)
        if size == -1:
            # watch out! size > 0x7FFFFFFF
            data = await self.reader.readexactly(8)
            (size,) = struct.unpack("!Q", data)
        data = await self.reader.readexactly(size)
        return pickle.loads(data)

    async def send(self, message: ChildMsg) -> None:
        data = pickle.dumps(message)
        size = len(data)
        if size < 2**31:
            header = struct.pack("!i", size)
        else:
            header = struct.pack("!iQ", -1, size)
        self.writer.write(header)
        self.writer.write(data)
        await self.writer.drain()


class ChildServer:
    def __init__(self, num_processes: int) -> None:
        self.conns: list[AsyncConnection] = []
        self.connected = asyncio.Condition()
        self.queue: asyncio.Queue[ChildMsg] = asyncio.Queue()
        self.num_processes = num_processes

    async def child_connected_cb(self, r: StreamReader, w: StreamWriter) -> None:
        c = AsyncConnection(r, w)
        self.conns.append(c)
        async with self.connected:
            self.connected.notify_all()
        while not SHUTDOWN_NOW.exists():
            try:
                msg = await c.recv()
            except (asyncio.exceptions.IncompleteReadError, EOFError) as e:
                if SHUTDOWN_NOW.exists() or SHUTDOWN_REQUESTED.exists():
                    logger.info("Child process disconnected during graceful shutdown")
                else:
                    logger.error(
                        f"Child process unexpectedly disconnected. Did something terrible happen? {repr(e)}"
                    )
                    FATAL_ERROR.touch()
                    exc = Exception("Worker crashed for this input!")
                    msg = ChildMsg(-1, Message.ERROR, exc)
                    await self.queue.put(msg)
                return
            await self.queue.put(msg)

    async def start(self, sprocket_socket: str) -> None:
        await asyncio.start_unix_server(self.child_connected_cb, path=sprocket_socket)

    async def wait_for_connections(self) -> None:
        async with self.connected:
            await self.connected.wait_for(lambda: len(self.conns) == self.num_processes)

    async def broadcast(self, msg: Any) -> None:
        for conn in self.conns:
            await conn.send(msg)

    async def gather(self) -> AsyncIterator[ChildMsg]:
        while not SHUTDOWN_NOW.exists():
            yield await self.queue.get()


class TorchRunSprocket(AsyncSprocket):
    async def setup(self) -> None:
        sprocket_socket = f"/tmp/sprocket-{os.getpid()}"
        self.num_processes = int(os.getenv("WORLD_SIZE", "1"))
        torchrun_args = [
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={self.num_processes}",
        ]
        os.environ["SPROCKET_SOCKET"] = sprocket_socket

        self.child_server = ChildServer(self.num_processes)
        await self.child_server.start(sprocket_socket)

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
        await self.child_server.wait_for_connections()
        setup_starts = 0
        setup_dones = 0
        print("Awaiting worker setup")
        async for msg in self.child_server.gather():
            match msg.type:
                case Message.SETUP_START:
                    setup_starts += 1
                case Message.SETUP_DONE:
                    setup_dones += 1
                case Message.SETUP_ERROR:
                    raise msg.arg
                case _:
                    raise ValueError(f"invalid msg {msg}")
            if setup_starts == setup_dones == self.num_processes:
                break
        print("Worker setup complete")

    async def predict(self, args: dict) -> dict:
        await self.child_server.broadcast(args)
        predict_dones = 0
        result = None
        err = None
        async for msg in self.child_server.gather():
            match msg.type:
                case Message.ERROR:
                    err = msg.arg  # we will raise the last error we receive
                    if FATAL_ERROR.exists():
                        # one of the ranks crashed
                        # this job should not be retried and we need to restart immediately
                        raise err
                case Message.OUTPUT:
                    if msg.arg:  # ignore ranks returning None
                        result = msg.arg
                case Message.PREDICT_DONE:
                    predict_dones += 1
            if predict_dones == self.num_processes:
                if err:
                    raise err
                if result:
                    return result
                raise Exception("all ranks returned None")
        raise Exception("shutting down")

    async def shutdown(self) -> None:
        # todo: timeout
        if self.torchrun_proc:
            print("Killing torchrun")
            self.torchrun_proc.terminate()
            await self.torchrun_proc.wait()


PROCESS_START_TIME = time.time()


def run(sprocket: Sprocket, name: str, use_torchrun: bool = False) -> None:
    SHUTDOWN_NOW.unlink(missing_ok=True)
    SHUTDOWN_REQUESTED.unlink(missing_ok=True)
    FATAL_ERROR.unlink(missing_ok=True)

    if local_rank := os.getenv("LOCAL_RANK"):
        ChildRunner(sprocket, name, int(local_rank)).run()
    elif use_torchrun:
        # sprocket is ignored in the parent process, TorchRunSprocket handles launching subprocesses that use the real sprocket
        # copy over the processor and warmup inputs to the sprocket that will actually be used
        TorchRunSprocket.processor = sprocket.processor
        TorchRunSprocket.warmup_inputs = sprocket.warmup_inputs
        asyncio.run(Runner(TorchRunSprocket(), name).run())
    else:
        asyncio.run(Runner(sprocket, name).run())
