import asyncio
import logging
import os
import random
import socket
import time
from typing import Optional

import aiohttp

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


class Sprocket:
    def setup(self) -> None:
        raise NotImplementedError

    def predict(self, args: dict) -> dict:
        raise NotImplementedError


class Runner:
    session: aiohttp.ClientSession

    def __init__(self, sprocket: Sprocket, model_name: str) -> None:
        api_key = os.getenv("TOGETHER_API_KEY")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        if VERSION:
            self.headers["x-worker-version"] = VERSION
        self.shutdown_event = asyncio.Event()
        self.sprocket = sprocket
        self.model_name = model_name

    async def get_job(self) -> dict:
        params = {"timeout": "5s", "model": self.model_name, "hostname": HOSTNAME}
        start_time = time.time()
        while not self.shutdown_event.is_set():
            try:
                logger.info("waiting for job from remote queue")
                headers = {"x-idle-time": str(time.time() - start_time)}
                async with self.session.get(
                    RETRIEVE_URL, headers=headers, params=params
                ) as response:
                    response.raise_for_status()
                    resp = await response.json()
                    if resp.get("data"):
                        return resp["data"]
                    logger.info("no job found in remote queue")
            except aiohttp.ClientError as e:
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
                resp = await self.session.post(UPDATE_URL, json=data)
                if resp.status == 200:
                    return True
                resp.raise_for_status()
            except aiohttp.ClientError as e:
                logger.error(f"failed to update job status: {e}")
        logger.error("failed to update job status after 3 retries")
        return False

    async def run_one_job(self) -> None:
        job = await self.get_job()
        request_id = job["request_id"]
        try:
            output = self.sprocket.predict(job["payload"])
        except Exception as e:
            logger.error(f"Job {request_id} failed")
            await self.update_job_status(request_id, "failed", info={"error": str(e)})
        else:
            logger.info(f"Job {request_id} finished")
            await self.update_job_status(request_id, "done", outputs=output)

    async def run(self) -> None:
        async with aiohttp.ClientSession(headers=self.headers) as self.session:
            self.sprocket.setup()
            while not self.shutdown_event.is_set():
                await self.run_one_job()
