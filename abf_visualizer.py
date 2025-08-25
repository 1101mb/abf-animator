#!/usr/bin/env python3
"""
ABF moving-window animation (right-edge cursor; past to the left):
 - View window is x in [-WINDOW_SECONDS, 0], so the trace "starts" at the right edge (x=0)
 - No axes
 - Only the LEFT side (x <= 0) of the trace is drawn; right of 0 is blank
 - All settings are configured by constants below (no CLI flags needed)
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from moviepy.editor import VideoClip
import pyabf

# =========================
# User-configurable settings
# =========================
# If this file exists, it's used. Otherwise we auto-pick the first *.abf in the folder.
DEFAULT_ABF_PATH = r"C:\Users\user\Desktop\final1.abf"

# Playback speed multiplier (higher = faster scroll)
PLAYBACK_SPEED = 0.25

# Total window width in seconds (e.g., 1.0 = last 1 s shown across the frame)
WINDOW_SECONDS = 1.0

# Frame rate and output size
FPS = 60
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

# Plot look
LINE_WIDTH = 1.5
SHOW_CENTER_LINE = False         # no center line
BACKGROUND_COLOR = "white"       # "white" or "black"
TRACE_COLOR = "black"            # trace line color
CENTER_LINE_COLOR = None         # unused since SHOW_CENTER_LINE=False
SWEEP_INDEX = 0                  # which sweep to animate (0-based)
# =========================


def resolve_abf_path() -> str:
    """Return an ABF path: DEFAULT_ABF_PATH if it exists, else first *.abf nearby, else error."""
    if DEFAULT_ABF_PATH and os.path.isfile(DEFAULT_ABF_PATH):
        return os.path.abspath(DEFAULT_ABF_PATH)

    # Try current working dir
    matches = sorted(glob.glob(os.path.join(os.getcwd(), "*.abf")))
    if matches:
        return os.path.abspath(matches[0])

    # Try the folder where this script sits
    script_dir = os.path.dirname(os.path.abspath(__file__))
    matches = sorted(glob.glob(os.path.join(script_dir, "*.abf")))
    if matches:
        return os.path.abspath(matches[0])

    raise SystemExit("No ABF file found: DEFAULT_ABF_PATH is missing and no *.abf in this folder.")


def build_clip(abf_path: str,
               out_path: str,
               sweep: int = 0,
               window_s: float = 1.0,
               fps: int = 60,
               speed: float = 1.0,
               width: int = 1280,
               height: int = 720,
               line_width: float = 1.5):
    """
    Moving-window animation with a right-edge "cursor" at x=0.
    Visible window is [-window_s, 0]; only x <= 0 is drawn.
    """
    # --- Load ABF data ---
    abf = pyabf.ABF(abf_path)
    if sweep < 0 or sweep >= abf.sweepCount:
        raise ValueError(f"sweep index {sweep} out of range [0, {abf.sweepCount-1}]")
    abf.setSweep(sweep)
    t = np.asarray(abf.sweepX, dtype=float)  # seconds
    y = np.asarray(abf.sweepY, dtype=float)  # mV/pA/etc.

    total_data_duration = float(t[-1] - t[0])
    if total_data_duration <= 0:
        raise RuntimeError("ABF time vector has non-positive duration.")

    # Video duration scales with speed
    speed = max(float(speed), 1e-6)
    duration_video = total_data_duration / speed

    # --- Y-limits (full data range + small margin) ---
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    margin = 0.05 * (y_max - y_min + 1e-6)   # 5% margin
    y_min -= margin
    y_max += margin

    # --- Matplotlib figure (Agg canvas) ---
    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)          # fill the whole frame
    ax.set_axis_off()
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    FigureCanvas(fig)  # ensure Agg canvas

    # Right-edge cursor design: window is [-window_s, 0]
    win = float(window_s)
    ax.set_xlim(-win, 0.0)
    ax.set_ylim(y_min, y_max)

    # Artists
    (trace_line,) = ax.plot([], [], lw=line_width)
    if TRACE_COLOR:
        trace_line.set_color(TRACE_COLOR)
    if SHOW_CENTER_LINE:
        cl = ax.axvline(0.0, lw=1.0)
        if CENTER_LINE_COLOR:
            cl.set_color(CENTER_LINE_COLOR)
    trace_line.set_clip_on(True)

    # Pre-cast for faster slicing
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    def make_frame(video_time: float):
        """
        Called per frame by MoviePy at 'video_time' (s). Map to data time 'center_t' = video_time * speed.
        The right edge (x=0) corresponds to center_t; we show [center_t - win, center_t].
        """
        center_t = video_time * speed
        left_t = center_t - win
        right_t = center_t

        i0 = np.searchsorted(t, left_t, side="left")
        i1 = np.searchsorted(t, right_t, side="right")
        i0 = max(0, i0)
        i1 = min(len(t), i1)

        if i1 <= i0 + 1:
            tt_rel = np.array([-win, 0.0], dtype=float)
            yy = np.array([np.nan, np.nan], dtype=float)
        else:
            # Shift so the right edge is x=0; clip to <= 0 (left side only)
            tt_rel = t[i0:i1] - center_t
            yy = y[i0:i1]
            mask = tt_rel <= 0.0
            tt_rel = tt_rel[mask]
            yy = yy[mask]

            if tt_rel.size == 0:
                tt_rel = np.array([-win, 0.0], dtype=float)
                yy = np.array([np.nan, np.nan], dtype=float)

        trace_line.set_data(tt_rel, yy)

        # Render to RGBA, convert to RGB for MoviePy
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())  # H, W, 4 (uint8)
        frame = buf[..., :3].copy()
        return frame

    clip = VideoClip(make_frame, duration=duration_video).set_fps(fps)
    clip.write_videofile(out_path, codec="libx264", fps=fps, audio=False, verbose=False, logger=None)
    plt.close(fig)


def main():
    abf_path = resolve_abf_path()
    stem = os.path.splitext(os.path.basename(abf_path))[0]
    out_path = os.path.join(os.path.dirname(abf_path), f"{stem}_center_scroll.mp4")

    build_clip(
        abf_path=abf_path,
        out_path=out_path,
        sweep=SWEEP_INDEX,
        window_s=WINDOW_SECONDS,
        fps=FPS,
        speed=PLAYBACK_SPEED,
        width=VIDEO_WIDTH,
        height=VIDEO_HEIGHT,
        line_width=LINE_WIDTH,
    )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
