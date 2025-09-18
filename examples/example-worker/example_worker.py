import time
import asyncio
from sprocket import Sprocket, Runner


class ExampleSprocket(Sprocket):
    def setup(self) -> None:
        self.multiplier = 2

    def predict(self, args: dict) -> dict:
        time.sleep(float(args.get("sleep", 0.5)))
        value = args["multiplicand"] * self.multiplier
        return {"value": value}


if __name__ == "__main__":
    asyncio.run(Runner(ExampleSprocket(), "sprocket-queue-test").run())
