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
from pathlib import Path

import sprocket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PANTRY = Path("/pantry")


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
    queue_name = os.environ.get("TOGETHER_DEPLOYMENT_NAME", "kitchen-sink")
    sprocket.run(KitchenSink(), queue_name)
