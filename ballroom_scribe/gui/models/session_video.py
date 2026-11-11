"""Type alias for an entry of the editor's video session: a loaded video or an offline one."""

from .loaded_video import LoadedVideo
from .offline_video import OfflineVideo

type SessionVideo = LoadedVideo | OfflineVideo
