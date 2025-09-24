import asyncio
from pathlib import Path
import pickle
import struct
from typing import Any, AsyncIterator, Generic, TypeVar



class AsyncConnection:
    def __init__(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.reader = reader
        self.writer = writer

    async def recv(self) -> Any:
        # this might need to be in an asyncio.gather or have a timeout to exit cleanly
        data = await self.reader.readexactly(4)
        (size,) = struct.unpack("!i", data)
        if size == -1:
            # watch out! size > 0x7FFFFFFF
            data = await self.reader.readexactly(8)
            (size,) = struct.unpack("!Q", data)
        data = await self.reader.readexactly(size)
        return pickle.loads(data)

    async def send(self, message: Any) -> None:
        data = pickle.dumps(message)
        size = len(data)
        if size < 2**31:
            header = struct.pack("!i", size)
        else:
            header = struct.pack("!iQ", -1, size)
        self.writer.write(header)
        self.writer.write(data)
        await self.writer.drain()

T = TypeVar("T")

class ConnectionManager(Generic[T]):
    def __init__(self, *, shutdown_path: Path) -> None:
        self.conns: list[AsyncConnection] = []
        self.shutdown_path = shutdown_path
        self.connected = asyncio.Condition()
        self.queue: asyncio.Queue[T] = asyncio.Queue()

    async def run(self, sprocket_socket: str) -> None:
        async def start_connection(
            r: asyncio.StreamReader, w: asyncio.StreamWriter
        ) -> None:
            c = AsyncConnection(r, w)
            self.conns.append(c)
            async with self.connected:
                self.connected.notify_all()
            while not self.shutdown_path.exists():
                await self.queue.put(await c.recv())

        self.listener = await asyncio.start_unix_server(
            start_connection, path=sprocket_socket
        )

    async def wait_for_connections(self, num_processes: int) -> None:
        async with self.connected:
            await self.connected.wait_for(lambda: len(self.conns) == num_processes)

    async def broadcast(self, msg: Any) -> None:
        for conn in self.conns:
            await conn.send(msg)

    async def gather(self) -> "AsyncIterator[T]":
        while not self.shutdown_path.exists():
            yield await self.queue.get()
