# streaming/mjpeg_server.py
from __future__ import annotations
import asyncio
from io import BytesIO
from typing import AsyncIterator, Callable, Any

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from PIL import Image

from core.base_env import Env
from policies.actor_critic import ActorCritic


class FrameBuffer:
    """
    Holds the latest frame as JPEG. Producer overwrites, consumers stream.
    """
    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)

    async def push(self, jpeg_bytes: bytes) -> None:
        if self._queue.full():
            try:
                _ = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(jpeg_bytes)

    async def stream(self) -> AsyncIterator[bytes]:
        boundary = b"--frame"
        while True:
            frame = await self._queue.get()
            yield (
                boundary
                + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """
    frame: H x W x 3, uint8 RGB
    """
    img = Image.fromarray(frame)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# Type aliases
EnvFactory = Callable[[], Env]
PolicyFactory = Callable[[Any], ActorCritic]


def create_app(
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    checkpoint_path: str,
    device: str = "cpu",
) -> FastAPI:
    """
    Returns a FastAPI app that:
    - On startup: runs eval loop in background
    - Exposes /stream endpoint for MJPEG
    """
    app = FastAPI()
    frame_buffer = FrameBuffer()

    async def eval_loop():
        env = env_factory()
        policy = policy_factory(env.spec).to(device)
        policy.eval()

        state_dict = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(state_dict)

        obs = env.reset()
        try:
            while True:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_t, _ = policy.act(obs_t, deterministic=True)
                action = action_t.squeeze(0).cpu().numpy()

                step_res = env.step(action)
                obs = step_res.obs if not step_res.done else env.reset()

                if step_res.frame is not None:
                    jpeg = encode_jpeg(step_res.frame)
                    await frame_buffer.push(jpeg)

                # yield back to the event loop
                await asyncio.sleep(0)
        finally:
            env.close()

    @app.on_event("startup")
    async def _startup_event():
        asyncio.create_task(eval_loop())

    @app.get("/stream")
    async def stream():
        return StreamingResponse(
            frame_buffer.stream(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return """
        <html>
          <head><title>Walker Stream</title></head>
          <body style="background:#222; color:#eee; text-align:center;">
            <h1>Walker Policy Visualization</h1>
            <img src="/stream" style="max-width:90%; border:2px solid #555;" />
          </body>
        </html>
        """

    return app