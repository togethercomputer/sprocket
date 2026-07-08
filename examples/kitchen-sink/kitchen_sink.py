"""Kitchen-sink sprocket worker — exercises every platform feature.

- Volumes:    pantry mounted at /pantry with recipe files
- Secrets:    SECRET_SPICE env var injected from a secret
- File I/O:   read arbitrary paths, return content
- FileOutput: write a receipt to disk and return it for upload
- Queue:      process jobs async via queue consumer
- HTTP:       walk-up counter for sync requests
"""

import logging
import os
import tempfile
import time
import traceback
from pathlib import Path

import sprocket
from sprocket import sprocket as _sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PANTRY = Path("/pantry")


# --- diagnostics: instrument FileOutput uploads to pinpoint transient failures ---
# kitchen_sink.py is copied into the worker image directly (sprocket is
# pip-installed), so this ships on the next image build without a sprocket
# release. We wrap QueueClient.upload_file — which runs in THIS worker process,
# so it sees the *actual* failing request, not a re-connection — to record which
# hop broke (presigned POST vs Tigris PUT), at which layer (TCP connect vs TLS
# handshake), and the underlying cause, when uploads intermittently ConnectError.
_orig_upload_file = _sp.QueueClient.upload_file


def _cause_chain(exc: BaseException) -> str:
    chain, cur = [], exc.__cause__ or exc.__context__
    while cur is not None:
        chain.append(f"{type(cur).__module__}.{type(cur).__name__}({cur})")
        cur = cur.__cause__ or cur.__context__
    return " <- ".join(chain) or "(none)"


async def _instrumented_upload_file(self, request_id, path):
    start = time.time()
    try:
        return await _orig_upload_file(self, request_id, path)
    except Exception as e:
        # httpx attaches the in-flight request to the exception, so this is the
        # exact request that failed (full presigned URL → host + which hop).
        req = getattr(e, "request", None)
        failing = f"{req.method} {req.url}" if req is not None else "(no request on exception)"
        logger.error(
            "FileOutput upload FAILED after %.1fs for %s-%s\n"
            "  failing request: %s\n"
            "  error: %r\n"
            "  cause chain: %s\n%s",
            time.time() - start, request_id, path.name, failing,
            e, _cause_chain(e), traceback.format_exc(),
        )
        raise


_sp.QueueClient.upload_file = _instrumented_upload_file


class KitchenSink(sprocket.Sprocket):
    def setup(self) -> None:
        self.secret_spice = os.environ.get("SECRET_SPICE", "love")
        self.recipes: dict[str, str] = {}
        if PANTRY.exists():
            for recipe_file in PANTRY.glob("*.txt"):
                self.recipes[recipe_file.stem] = recipe_file.read_text().strip()
        logger.info("Kitchen ready — %d recipes, secret spice loaded", len(self.recipes))

    def predict(self, args: dict) -> dict:
        time.sleep(float(args.get("sleep", 0)))
        result: dict = {}

        if dish := args.get("dish"):
            recipe = self.recipes.get(dish, "freestyle")
            result["plate"] = f"Made {dish} using {recipe}, seasoned with {self.secret_spice}"

        if "menu" in args:
            result["menu"] = sorted(self.recipes.keys())

        if env_name := args.get("read_env"):
            result["env_value"] = os.environ.get(env_name)

        if file_path := args.get("read_file"):
            try:
                result["file_content"] = Path(file_path).read_text().strip()
            except Exception as e:
                result["file_error"] = str(e)

        if "receipt" in args:
            receipt = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
            receipt.write(f"Receipt: {args.get('dish', 'unknown')}, spice={self.secret_spice}\n".encode())
            receipt.close()
            result["receipt"] = sprocket.FileOutput(receipt.name)

        return result


if __name__ == "__main__":
    sprocket.run(KitchenSink())
