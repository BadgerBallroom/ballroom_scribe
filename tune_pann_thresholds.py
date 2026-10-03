"""Streamlit app for visually tuning PANN audio classification thresholds."""

import http.server
import json
import mimetypes
import os
import pathlib
import re
import threading
import urllib.parse

import numpy as np
import soundfile
import streamlit as st

import ballroom_scribe
import ballroom_scribe.cli.arguments
import ballroom_scribe.models
import ballroom_scribe.utils

_RANGE_HEADER_RE = re.compile(r"bytes=(\d+)-(\d*)")
_STREAM_CHUNK_BYTES = 1024 * 1024  # 1 MB


@st.cache_data(show_spinner="Loading or computing PANN framewise probabilities...")
def load_probabilities(
    input_video: pathlib.Path,
    input_channel: int,
    cache_dir: pathlib.Path,
    pann_device: str,
    pann_chunk_minutes: float,
    pann_chunk_overlap_seconds: float,
) -> tuple[np.typing.NDArray[np.float32], float]:
    """Loads or computes stitched framewise probabilities and total audio duration in seconds."""
    cache_dir = ballroom_scribe.utils.cache_dir_for_video(
        parent_cache_dir=cache_dir,
        input_video=input_video,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    # `extract_audio` is idempotent and returns immediately if the audio has already been extracted.
    with st.spinner(f"Extracting audio from {input_video.name} (this may take a few minutes for large files)..."):
        extracted_audio = ballroom_scribe.utils.extract_audio(
            input_video=input_video,
            input_channel=input_channel,
            cache_dir=cache_dir,
        )

    audio_loader = ballroom_scribe.AudioLoader(
        chunk_minutes=pann_chunk_minutes,
        chunk_overlap_seconds=pann_chunk_overlap_seconds,
    )

    pann_inferencer: ballroom_scribe.PannInferencer | None = ballroom_scribe.PannInferencer(device=pann_device)
    try:
        unclassified_chunks = audio_loader.load(input_audio=extracted_audio)
        probabilities = pann_inferencer.infer(
            audio_chunks=unclassified_chunks,
            channel=input_channel,
            cache_dir=cache_dir,
        )
    finally:
        del pann_inferencer
        ballroom_scribe.utils.free_vram()

    duration_seconds = float(soundfile.info(extracted_audio).duration)
    return np.ascontiguousarray(probabilities, dtype=np.float32), duration_seconds


@st.cache_resource
def start_local_media_server(
    video_path: pathlib.Path,
    _probabilities: np.typing.NDArray[np.float32],
) -> int:
    """Starts a background HTTP server on all interfaces and returns its port."""
    video_size = video_path.stat().st_size
    video_mime, _ = mimetypes.guess_type(video_path.name)
    video_content_type = video_mime or "video/mp4"
    probs_bytes = memoryview(_probabilities).cast("B")

    class _MediaRangeHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass  # Silence per-chunk HTTP console logs

        def do_HEAD(self) -> None:
            self._handle_request(send_body=False)

        def do_GET(self) -> None:
            self._handle_request(send_body=True)

        def _handle_request(self, *, send_body: bool) -> None:
            if self.path.startswith("/probabilities.bin"):
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(probs_bytes)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                if send_body:
                    try:
                        self.wfile.write(probs_bytes)
                    except BrokenPipeError, ConnectionAbortedError, ConnectionResetError:
                        pass
                return

            if not self.path.startswith("/video"):
                self.send_error(404)
                return

            range_header = self.headers.get("Range")
            start = 0
            end = video_size - 1

            if range_header:
                match = _RANGE_HEADER_RE.search(range_header)
                if match:
                    start = int(match.group(1))
                    if match.group(2):
                        end = min(video_size - 1, int(match.group(2)))
                if start >= video_size:
                    self.send_error(416, "Requested Range Not Satisfiable")
                    return
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{video_size}")
            else:
                self.send_response(200)

            content_length = end - start + 1
            self.send_header("Content-Type", video_content_type)
            self.send_header("Content-Length", str(content_length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if not send_body:
                return

            try:
                with video_path.open("rb") as f:
                    f.seek(start, os.SEEK_SET)
                    remaining = content_length
                    while remaining > 0:
                        chunk = f.read(min(_STREAM_CHUNK_BYTES, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except BrokenPipeError, ConnectionAbortedError, ConnectionResetError:
                # Browser routinely aborts open Range connections when seeking
                pass

    # Bind to 0.0.0.0 so other devices on the LAN can connect
    server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), _MediaRangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    return server.server_port


def _build_synchronized_plot_html(
    probs_url: str,
    num_frames: int,
    num_classes: int,
    duration_seconds: float,
    class_names: list[str],
    thresholds: list[float],
    window_seconds: float,
) -> str:
    """Builds the client-side HTML/JS canvas plot synchronized with Streamlit's <video> element."""
    config_json = json.dumps(
        {
            "probsUrl": probs_url,
            "numFrames": num_frames,
            "numClasses": num_classes,
            "durationSeconds": duration_seconds,
            "classNames": class_names,
            "thresholds": thresholds,
            "initialWindowSeconds": window_seconds,
        }
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: #0e1117;
                color: #fafafa;
                font-family: sans-serif;
                user-select: none;
            }}
            #toolbar {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 6px 10px;
                font-size: 13px;
                background: #161b22;
                border-bottom: 1px solid #30363d;
            }}
            .legend-item {{
                display: inline-flex;
                align-items: center;
                margin-right: 16px;
                gap: 6px;
            }}
            .swatch {{
                width: 12px;
                height: 12px;
                border-radius: 2px;
                display: inline-block;
            }}
            canvas {{
                width: 100%;
                height: 280px;
                display: block;
                cursor: ew-resize;
            }}
        </style>
    </head>
    <body>
        <div id="toolbar">
            <div id="legend"></div>
            <div id="readout">Loading probabilities...</div>
        </div>
        <canvas id="plot"></canvas>

        <script>
            const CONFIG = {config_json};
            const COLORS = ["#3b82f6", "#10b981", "#f59e0b"]; // Speech (blue), Music (green), Applause (amber)
            const OTHER_COLOR = "#374151";

            let probs = null;
            let windowSeconds = CONFIG.initialWindowSeconds;
            const fps = CONFIG.numFrames / CONFIG.durationSeconds;

            // Populate legend
            const legendEl = document.getElementById("legend");
            CONFIG.classNames.forEach((name, idx) => {{
                const item = document.createElement("span");
                item.className = "legend-item";
                item.innerHTML = `<span class="swatch" style="background:${{COLORS[idx]}}"></span>` +
                                 `${{name}} (thresh: ${{CONFIG.thresholds[idx].toFixed(2)}})`;
                legendEl.appendChild(item);
            }});

            const canvas = document.getElementById("plot");
            const ctx = canvas.getContext("2d");
            const readoutEl = document.getElementById("readout");

            function getParentVideo() {{
                try {{
                    return window.parent.document.querySelector("video");
                }} catch (e) {{
                    return null;
                }}
            }}

            function classifyFrame(frameIdx) {{
                const base = frameIdx * CONFIG.numClasses;
                for (let c = 0; c < CONFIG.numClasses; c++) {{
                    if (probs[base + c] > CONFIG.thresholds[c]) {{
                        return c;
                    }}
                }}
                return -1; // OTHER
            }}

            function formatTime(sec) {{
                const s = Math.max(0, sec);
                const h = Math.floor(s / 3600);
                const m = Math.floor((s % 3600) / 60).toString().padStart(2, "0");
                const rem = (s % 60).toFixed(2).padStart(5, "0");
                return h > 0 ? `${{h}}:${{m}}:${{rem}}` : `${{m}}:${{rem}}`;
            }}

            function draw(currentTime) {{
                if (!probs) return;

                const dpr = window.devicePixelRatio || 1;
                const rect = canvas.getBoundingClientRect();
                if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {{
                    canvas.width = Math.round(rect.width * dpr);
                    canvas.height = Math.round(rect.height * dpr);
                }}
                ctx.save();
                ctx.scale(dpr, dpr);

                const W = rect.width;
                const H = rect.height;
                const stripH = 18;
                const padTop = stripH + 10;
                const padBottom = 24;
                const plotH = H - padTop - padBottom;

                ctx.fillStyle = "#0e1117";
                ctx.fillRect(0, 0, W, H);

                const halfWin = windowSeconds / 2;
                const startSec = currentTime - halfWin;
                const endSec = currentTime + halfWin;

                const startFrame = Math.max(0, Math.floor(startSec * fps));
                const endFrame = Math.min(CONFIG.numFrames - 1, Math.ceil(endSec * fps));
                const step = Math.max(1, Math.floor((endFrame - startFrame) / W));

                // 1. Draw winning classification strip at the top
                for (let f = startFrame; f <= endFrame; f += step) {{
                    const t = f / fps;
                    const x0 = ((t - startSec) / windowSeconds) * W;
                    const x1 = (((f + step) / fps - startSec) / windowSeconds) * W;
                    const winClass = classifyFrame(f);
                    ctx.fillStyle = winClass >= 0 ? COLORS[winClass] : OTHER_COLOR;
                    ctx.fillRect(x0, 0, Math.max(1, x1 - x0 + 0.5), stripH);
                }}

                // 2. Draw horizontal grid & dashed threshold lines
                ctx.lineWidth = 1;
                for (let c = 0; c < CONFIG.numClasses; c++) {{
                    const y = padTop + (1 - CONFIG.thresholds[c]) * plotH;
                    ctx.strokeStyle = COLORS[c];
                    ctx.setLineDash([5, 5]);
                    ctx.beginPath();
                    ctx.moveTo(0, y);
                    ctx.lineTo(W, y);
                    ctx.stroke();
                }}
                ctx.setLineDash([]);

                // 3. Draw probability curves
                for (let c = 0; c < CONFIG.numClasses; c++) {{
                    ctx.strokeStyle = COLORS[c];
                    ctx.lineWidth = 1.75;
                    ctx.beginPath();
                    let started = false;
                    for (let f = startFrame; f <= endFrame; f += step) {{
                        const t = f / fps;
                        const x = ((t - startSec) / windowSeconds) * W;
                        const p = probs[f * CONFIG.numClasses + c];
                        const y = padTop + (1 - p) * plotH;
                        if (!started) {{
                            ctx.moveTo(x, y);
                            started = true;
                        }} else {{
                            ctx.lineTo(x, y);
                        }}
                    }}
                    ctx.stroke();
                }}

                // 4. Draw center playhead
                const centerX = W / 2;
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(centerX, 0);
                ctx.lineTo(centerX, H - padBottom);
                ctx.stroke();

                // 5. Update readout for current frame
                const curFrame = Math.min(CONFIG.numFrames - 1, Math.max(0, Math.round(currentTime * fps)));
                if (CONFIG.numFrames > 0) {{
                    const winIdx = classifyFrame(curFrame);
                    const winName = winIdx >= 0 ? CONFIG.classNames[winIdx] : "OTHER";
                    const vals = CONFIG.classNames.map(
                        (name, c) => `${{name}}: ${{probs[curFrame * CONFIG.numClasses + c].toFixed(2)}}`
                    ).join(" | ");
                    readoutEl.textContent =
                        `t = ${{formatTime(currentTime)}} (win: ${{windowSeconds.toFixed(1)}}s) → ${{winName}} (${{vals}})`;
                }}

                ctx.restore();
            }}

            // Pointer drag / click to seek video
            let isDragging = false;
            function seekFromEvent(e) {{
                const video = getParentVideo();
                if (!video) return;
                const rect = canvas.getBoundingClientRect();
                const relX = (e.clientX - rect.left) / rect.width - 0.5;
                video.currentTime = Math.max(0, Math.min(CONFIG.durationSeconds, video.currentTime + relX * windowSeconds));
            }}
            canvas.addEventListener("mousedown", (e) => {{ isDragging = true; seekFromEvent(e); }});
            window.addEventListener("mousemove", (e) => {{ if (isDragging) seekFromEvent(e); }});
            window.addEventListener("mouseup", () => {{ isDragging = false; }});

            // Scroll wheel on canvas to adjust zoom window smoothly
            canvas.addEventListener("wheel", (e) => {{
                e.preventDefault();
                const factor = e.deltaY > 0 ? 1.15 : 1 / 1.15;
                windowSeconds = Math.max(2.0, Math.min(CONFIG.durationSeconds, windowSeconds * factor));
            }}, {{ passive: false }});

            function loop() {{
                const video = getParentVideo();
                const currentTime = video ? video.currentTime : 0.0;
                draw(currentTime);
                requestAnimationFrame(loop);
            }}

            fetch(CONFIG.probsUrl)
                .then((res) => res.arrayBuffer())
                .then((buf) => {{
                    probs = new Float32Array(buf);
                    requestAnimationFrame(loop);
                }})
                .catch((err) => {{
                    readoutEl.textContent = "Failed to load probabilities: " + err;
                }});
        </script>
    </body>
    </html>
    """


def main() -> None:
    st.set_page_config(page_title="PANN Threshold Tuner", layout="wide")
    input_cache_args, audio_classification_args = ballroom_scribe.cli.arguments.parse_args(
        (
            ballroom_scribe.cli.arguments.InputCacheGroup(),
            ballroom_scribe.cli.arguments.AudioClassificationGroup(),
        ),
        description="Visually tune PANN audio classification thresholds alongside video playback.",
    )

    st.title(f"PANN Threshold Tuner — `{input_cache_args['input_video'].name}`")

    # Sidebar controls initialized from CLI flags
    st.sidebar.header("Plot Controls")
    window_seconds = st.sidebar.slider(
        "Initial Window Width (seconds)",
        min_value=2.0,
        max_value=300.0,
        value=30.0,
        step=2.0,
        help="You can also zoom smoothly by scrolling the mouse wheel directly over the plot.",
    )

    st.sidebar.header("PANN Thresholds")
    speech_threshold = st.sidebar.slider(
        "Speech Threshold",
        min_value=0.0,
        max_value=1.0,
        value=audio_classification_args["pann_thresholds"][ballroom_scribe.models.AudioChunk.Classification.SPEECH],
        step=0.01,
    )
    music_threshold = st.sidebar.slider(
        "Music Threshold",
        min_value=0.0,
        max_value=1.0,
        value=audio_classification_args["pann_thresholds"][ballroom_scribe.models.AudioChunk.Classification.MUSIC],
        step=0.01,
    )

    st.sidebar.subheader("CLI Flags for `find_timestamps.py`")
    st.sidebar.code(
        f"--pann-threshold-speech {speech_threshold:.2f} \\\n--pann-threshold-music {music_threshold:.2f} \\\n",
        language="bash",
    )

    probabilities, duration_seconds = load_probabilities(
        input_video=input_cache_args["input_video"],
        input_channel=input_cache_args["input_channel"],
        cache_dir=input_cache_args["cache_dir"],
        pann_device=audio_classification_args["pann_device"],
        pann_chunk_minutes=audio_classification_args["pann_chunk_minutes"],
        pann_chunk_overlap_seconds=audio_classification_args["pann_chunk_overlap_seconds"],
    )

    media_port = start_local_media_server(
        video_path=input_cache_args["input_video"],
        _probabilities=probabilities,
    )

    # Use the exact hostname/IP that the current client browser used to reach Streamlit
    host_header = st.context.headers.get("Host", "localhost")
    media_host = urllib.parse.urlsplit(f"//{host_header}").hostname or "localhost"
    video_url = f"http://{media_host}:{media_port}/video"
    probs_url = f"http://{media_host}:{media_port}/probabilities.bin"

    threshold_map = {
        ballroom_scribe.models.AudioChunk.Classification.SPEECH: speech_threshold,
        ballroom_scribe.models.AudioChunk.Classification.MUSIC: music_threshold,
    }
    ordered_classifications = ballroom_scribe.PannInferencer.CLASSIFICATIONS
    class_names = [c.name for c in ordered_classifications]
    ordered_thresholds = [threshold_map[c] for c in ordered_classifications]

    # Passing an http:// URL bypasses Streamlit's in-memory file hashing and enables HTTP Range streaming
    st.video(video_url)

    # Synchronized moving-window plot below the video
    num_frames, num_classes = probabilities.shape
    plot_html = _build_synchronized_plot_html(
        probs_url=probs_url,
        num_frames=num_frames,
        num_classes=num_classes,
        duration_seconds=duration_seconds,
        class_names=class_names,
        thresholds=ordered_thresholds,
        window_seconds=window_seconds,
    )
    st.iframe(plot_html, height=325)


if __name__ == "__main__":
    main()
