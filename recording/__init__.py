# recording/__init__.py
from .video_recorder import (
    CameraSettings,
    RecordingConfig,
    record_video,
    record_video_simple,
)

__all__ = [
    "CameraSettings",
    "RecordingConfig",
    "record_video",
    "record_video_simple",
]