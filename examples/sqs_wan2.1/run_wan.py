import json
import os
import pathlib

import boto3
import requests
import torch
import torch.distributed as dist
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

import server

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_API_BASE_URL = os.getenv("TOGETHER_API_BASE_URL", "http://localhost:8081")
SQS_ENDPOINT_URL = f"{TOGETHER_API_BASE_URL}/internal/v1/queue/sqs/"
TEST_MODEL_NAME = "example-org/example-model-name"
QUEUE_URL = f"{SQS_ENDPOINT_URL}queue/{TEST_MODEL_NAME}"


def submit_job_for_testing():
    response = requests.post(
        f"{TOGETHER_API_BASE_URL}/v1/videos/generations",
        headers={
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": TEST_MODEL_NAME,
            "payload": {"prompt": "A cat sitting on a windowsill"},
            "priority": 1,
        },
    )
    response.raise_for_status()


class Worker:
    def setup(self) -> None:
        dist.init_process_group()
        torch.cuda.set_device(dist.get_rank())

        # if running in docker and not kubernetes, submit a job for testing
        if dist.get_rank() == 0 and pathlib.Path("/.dockerenv").exists():
            submit_job_for_testing()

        pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        self.pipe = pipe.to("cuda")

        para_mesh = init_context_parallel_mesh(self.pipe.device.type)
        parallelize_pipe(self.pipe, mesh=para_mesh)

        if dist.get_rank() == 0:
            self.sqs_client = boto3.client(
                "sqs",
                endpoint_url=SQS_ENDPOINT_URL,
                region_name="us-east-1",
                aws_access_key_id=TOGETHER_API_KEY,  # access key = Together API key
                aws_secret_access_key="dummy-secret",  # signature is not validated
            )
            server.run_health_and_metrics_server_in_background()

    def run_one_job(self) -> None:
        job = [None]
        receipt_handle = None
        if dist.get_rank() == 0:
            assert self.sqs_client
            response = self.sqs_client.receive_message(
                QueueUrl=QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=10
            )
            messages = response.get("Messages", [])
            if messages:
                message = messages[0]
                receipt_handle = message["ReceiptHandle"]
                job[0] = json.loads(message["Body"])["prompt"]
                server.set_utilization(busy=True)

        dist.broadcast_object_list(job, src=0)

        if job[0] is None:
            return

        video = self.pipe(
            prompt=job[0],
            negative_prompt="",
            height=480,
            width=832,
            num_frames=81,
            num_inference_steps=30,
            output_type="pil" if dist.get_rank() == 0 else "pt",
        ).frames[0]

        if dist.get_rank() == 0:
            assert self.sqs_client and receipt_handle
            print(f"Saving video to wan-{receipt_handle}.mp4")
            export_to_video(video, f"wan-{receipt_handle}.mp4", fps=15)
            # you would handle upload here
            # mark job as done by deleting message from queue
            self.sqs_client.delete_message(
                QueueUrl=QUEUE_URL, ReceiptHandle=receipt_handle
            )
            server.set_utilization(busy=False)

    def run(self):
        try:
            self.setup()
            while True:
                self.run_one_job()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    Worker().run()
