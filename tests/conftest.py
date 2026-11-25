from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def asgi_client():
    @asynccontextmanager
    async def client(app, *, peer=("127.0.0.1", 40000)):
        transport = httpx.ASGITransport(app=app, client=peer)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as connection:
                yield connection

    return client


def pytest_collection_modifyitems(items):
    for item in items:
        missing = [
            str(path)
            for mark in ("artifacts", "data")
            for marker in item.iter_markers(mark)
            for path in marker.args
            if not (ROOT / path).exists()
        ]
        if missing:
            item.add_marker(pytest.mark.skip(reason=f"Missing test inputs: {', '.join(missing)}"))
