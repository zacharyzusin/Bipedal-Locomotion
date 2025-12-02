# streaming/mjpeg_server.py
from __future__ import annotations
import asyncio
from io import BytesIO
from typing import AsyncIterator, Callable, Any, Deque, Mapping

from collections import deque
from dataclasses import dataclass, asdict

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image

from core.base_env import Env
from policies.actor_critic import ActorCritic

@dataclass
class StepStats:
    t: int
    reward: float
    reward_components: Mapping[str, float]
    done: bool

@dataclass
class CameraSettings:
    distance: float = 3.0
    azimuth: float = 90.0
    elevation: float = -20.0
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.5)

class CameraSettingsRequest(BaseModel):
    distance: float | None = None
    azimuth: float | None = None
    elevation: float | None = None
    lookat_x: float | None = None
    lookat_y: float | None = None
    lookat_z: float | None = None

class StreamingState:
    def __init__(self, maxlen: int = 500):
        self.history: Deque[StepStats] = deque(maxlen=maxlen)
        self.last: StepStats | None = None
        self.camera = CameraSettings()

    def add_step(self, stats: StepStats):
        self.last = stats
        self.history.append(stats)

    def to_dict(self) -> Dict[str, Any]:
        hist = [asdict(s) for s in self.history]
        return {
            "last": asdict(self.last) if self.last is not None else None,
            "history": hist,
        }

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
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> FastAPI:
    """
    Returns a FastAPI app that:
    - On startup: runs eval loop in background
    - Exposes /stream for MJPEG
    - Exposes /stats for reward/done history
    - Exposes /camera for camera controls
    """
    app = FastAPI()
    frame_buffer = FrameBuffer()
    state = StreamingState(maxlen=500)

    async def eval_loop():
        env = env_factory()
        policy = policy_factory(env)
        if hasattr(policy, "to"):
            policy = policy.to(device)
        if hasattr(policy, "eval"):
            policy.eval()

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location=device)
            if hasattr(policy, "load_state_dict"):
                policy.load_state_dict(state_dict)

        obs = env.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        try:
            while True:
                if hasattr(env, "set_camera"):
                    cam = state.camera
                    env.set_camera(
                        distance=cam.distance,
                        azimuth=cam.azimuth,
                        elevation=cam.elevation,
                        lookat=cam.lookat,
                    )

                with torch.no_grad():
                    action, _ = policy.act(obs, deterministic=True)

                action = action.squeeze(0).cpu().numpy()
                step_res = env.step(action)
                
                # --- handle done & reset policy too ---
                if step_res.done:
                    obs = env.reset()
                    if hasattr(policy, "reset"):
                        policy.reset()
                else:
                    obs = step_res.obs

                # --- update stats ---
                info = step_res.info
                stats = StepStats(
                    t=info.get("t", 0),
                    reward=float(step_res.reward),
                    reward_components=info.get("reward_components", {}),
                    done=bool(step_res.done),
                )
                state.add_step(stats)
                # --------------------

                if step_res.frame is not None:
                    jpeg = encode_jpeg(step_res.frame)
                    await frame_buffer.push(jpeg)

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

    @app.get("/stats")
    async def get_stats():
        return JSONResponse(state.to_dict())

    @app.get("/camera")
    async def get_camera():
        return JSONResponse(asdict(state.camera))

    @app.post("/camera")
    async def set_camera(req: CameraSettingsRequest):
        cam = state.camera
        if req.distance is not None:
            cam.distance = req.distance
        if req.azimuth is not None:
            cam.azimuth = req.azimuth
        if req.elevation is not None:
            cam.elevation = req.elevation
        if req.lookat_x is not None or req.lookat_y is not None or req.lookat_z is not None:
            cam.lookat = (
                req.lookat_x if req.lookat_x is not None else cam.lookat[0],
                req.lookat_y if req.lookat_y is not None else cam.lookat[1],
                req.lookat_z if req.lookat_z is not None else cam.lookat[2],
            )
        return JSONResponse(asdict(cam))

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return """
        <html>
        <head>
          <title>Walker Visualization</title>
          <style>
            body { background:#111; color:#eee; font-family:sans-serif; margin:0; }
            #container { display:flex; gap:20px; padding:20px; }
            #video { flex: 2; text-align:center; }
            #sidebar { flex: 1; }
            #controls { background:#222; padding:15px; border-radius:8px; margin-bottom:20px; }
            #controls h3 { margin-top:0; }
            .control-row { margin:10px 0; display:flex; align-items:center; }
            .control-row label { width:80px; }
            .control-row input[type=range] { flex:1; margin:0 10px; }
            .control-row span { width:60px; text-align:right; }
            canvas { background:#222; }
            button { background:#444; color:#eee; border:none; padding:8px 16px; 
                     cursor:pointer; border-radius:4px; margin:5px; }
            button:hover { background:#555; }
          </style>
        </head>
        <body>
          <div id="container">
            <div id="video">
              <h2>Walker</h2>
              <img src="/stream" style="max-width:100%; border:2px solid #555;" />
            </div>
            <div id="sidebar">
              <div id="controls">
                <h3>Camera Controls</h3>
                <div class="control-row">
                  <label>Distance</label>
                  <input type="range" id="distance" min="1" max="10" step="0.1" value="3">
                  <span id="distanceVal">3.0</span>
                </div>
                <div class="control-row">
                  <label>Azimuth</label>
                  <input type="range" id="azimuth" min="0" max="360" step="1" value="90">
                  <span id="azimuthVal">90°</span>
                </div>
                <div class="control-row">
                  <label>Elevation</label>
                  <input type="range" id="elevation" min="-90" max="90" step="1" value="-20">
                  <span id="elevationVal">-20°</span>
                </div>
                <div class="control-row">
                  <label>Look X</label>
                  <input type="range" id="lookatX" min="-5" max="5" step="0.1" value="0">
                  <span id="lookatXVal">0.0</span>
                </div>
                <div class="control-row">
                  <label>Look Y</label>
                  <input type="range" id="lookatY" min="-5" max="5" step="0.1" value="0">
                  <span id="lookatYVal">0.0</span>
                </div>
                <div class="control-row">
                  <label>Look Z</label>
                  <input type="range" id="lookatZ" min="0" max="3" step="0.1" value="0.5">
                  <span id="lookatZVal">0.5</span>
                </div>
                <div style="margin-top:15px;">
                  <button onclick="resetCamera()">Reset Camera</button>
                  <button onclick="followMode()">Follow Mode</button>
                  <button onclick="topView()">Top View</button>
                  <button onclick="sideView()">Side View</button>
                </div>
              </div>
              <div id="stats">
                <h3>Reward</h3>
                <canvas id="rewardCanvas" width="400" height="200"></canvas>
                <pre id="lastStats"></pre>
              </div>
            </div>
          </div>
          <script>
            const sliders = ['distance', 'azimuth', 'elevation', 'lookatX', 'lookatY', 'lookatZ'];
            
            async function updateCamera() {
              const body = {
                distance: parseFloat(document.getElementById('distance').value),
                azimuth: parseFloat(document.getElementById('azimuth').value),
                elevation: parseFloat(document.getElementById('elevation').value),
                lookat_x: parseFloat(document.getElementById('lookatX').value),
                lookat_y: parseFloat(document.getElementById('lookatY').value),
                lookat_z: parseFloat(document.getElementById('lookatZ').value),
              };
              await fetch('/camera', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body)
              });
            }

            function setupSlider(id, valId, suffix='') {
              const slider = document.getElementById(id);
              const valSpan = document.getElementById(valId);
              slider.addEventListener('input', () => {
                valSpan.textContent = parseFloat(slider.value).toFixed(1) + suffix;
                updateCamera();
              });
            }

            setupSlider('distance', 'distanceVal');
            setupSlider('azimuth', 'azimuthVal', '°');
            setupSlider('elevation', 'elevationVal', '°');
            setupSlider('lookatX', 'lookatXVal');
            setupSlider('lookatY', 'lookatYVal');
            setupSlider('lookatZ', 'lookatZVal');

            function setSliders(d, az, el, lx, ly, lz) {
              document.getElementById('distance').value = d;
              document.getElementById('distanceVal').textContent = d.toFixed(1);
              document.getElementById('azimuth').value = az;
              document.getElementById('azimuthVal').textContent = az + '°';
              document.getElementById('elevation').value = el;
              document.getElementById('elevationVal').textContent = el + '°';
              document.getElementById('lookatX').value = lx;
              document.getElementById('lookatXVal').textContent = lx.toFixed(1);
              document.getElementById('lookatY').value = ly;
              document.getElementById('lookatYVal').textContent = ly.toFixed(1);
              document.getElementById('lookatZ').value = lz;
              document.getElementById('lookatZVal').textContent = lz.toFixed(1);
              updateCamera();
            }

            function resetCamera() { setSliders(3, 90, -20, 0, 0, 0.5); }
            function topView() { setSliders(5, 0, -89, 0, 0, 0); }
            function sideView() { setSliders(4, 0, -10, 0, 0, 0.5); }
            function followMode() { setSliders(3, 180, -15, 0, 0, 0.5); }

            // Fetch initial camera state
            fetch('/camera').then(r => r.json()).then(cam => {
              setSliders(cam.distance, cam.azimuth, cam.elevation, 
                         cam.lookat[0], cam.lookat[1], cam.lookat[2]);
            });

            async function fetchStats() {
              const res = await fetch('/stats');
              const data = await res.json();
              updateStats(data);
            }

            function updateStats(data) {
              const ctx = document.getElementById('rewardCanvas').getContext('2d');
              ctx.clearRect(0, 0, 400, 200);

              const hist = data.history || [];
              if (hist.length === 0) return;

              const rewards = hist.map(s => s.reward);
              const maxR = Math.max(...rewards);
              const minR = Math.min(...rewards);

              const n = rewards.length;
              for (let i = 1; i < n; i++) {
                const x0 = (i-1) / (n-1) * 400;
                const x1 = i / (n-1) * 400;
                const y0 = 200 - ((rewards[i-1] - minR) / (maxR - minR + 1e-6)) * 200;
                const y1 = 200 - ((rewards[i]   - minR) / (maxR - minR + 1e-6)) * 200;

                ctx.beginPath();
                ctx.moveTo(x0, y0);
                ctx.lineTo(x1, y1);
                ctx.strokeStyle = '#0f0';
                ctx.stroke();
              }

              const last = data.last;
              document.getElementById('lastStats').textContent =
                JSON.stringify(last, null, 2);
            }

            setInterval(fetchStats, 200);
          </script>
        </body>
        </html>
        """

    return app