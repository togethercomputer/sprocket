import asyncio
import logging
import time

from sprocket.sprocket import Runner, Sprocket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExampleSprocket(Sprocket):
    def setup(self) -> None:
        self.multiplier = 2

        for i in range(20):
            time.sleep(1)
            logger.info("Setup in progress... %i", i)

    def predict(self, args: dict) -> dict:
        time.sleep(float(args.get("sleep", 0.5)))
        value = args["multiplicand"] * self.multiplier
        return {"value": value}


if __name__ == "__main__":
    asyncio.run(Runner(ExampleSprocket(), "sprocket-queue-test").run())
