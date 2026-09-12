# BlinkTrack Analytics & Fatigue Monitor

A real-time webcam blink-tracking dashboard built with Streamlit, OpenCV, and
MediaPipe's Tasks Vision `FaceLandmarker`. Tracks blink rate, duration, and
staring streaks over a timed session, scores your "Blink Performance Index",
flags fatigue/eye-strain risk factors, and exports the full session log.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

The `FaceLandmarker` `.task` model bundle is downloaded automatically into
`./models/` on first run (needs an internet connection once).

## Usage

1. In the sidebar, set your session length (1–60 min) and, if needed, tune
   the EAR blink-sensitivity threshold.
2. (Optional) Enable **Claude sports-analyst commentary** and paste an
   Anthropic API key, or set the `ANTHROPIC_API_KEY` environment variable
   beforehand.
3. Flip **Start Session** to begin. The webcam feed shows a live EAR reading,
   a "BLINKING" indicator, and a "Face Not Detected" warning if tracking is
   lost.
4. When the timer runs out, the session ends automatically, 🎈 balloons fire,
   and a full report appears below: performance grade, diagnostics, optional
   AI commentary, and a CSV download of every blink event.

## Files

| File | Purpose |
|---|---|
| `blink_detector.py` | MediaPipe FaceLandmarker wrapper, EAR calculation, blink state machine |
| `blink_stats.py` | Rolling session metrics (BPM, durations, streaks, consistency/rhythm scores) |
| `performance_index.py` | Composite 0–100 score + A+–F grade |
| `ai_report.py` | Rule-based diagnostics + template report + optional Claude commentary |
| `app.py` | Streamlit UI and video processing loop |

## Notes

- Camera backend is auto-detected: `cv2.CAP_DSHOW` on Windows (reduces lag),
  the OS default elsewhere (macOS/Linux), with a plain fallback if the
  preferred backend fails to open.
- This is a local desktop app — it needs direct access to a webcam via
  OpenCV, so it must be run on the machine with the camera, not accessed
  as a static hosted web page.
- AI commentary requires the `anthropic` package and a valid API key; without
  one, the app still produces the full template-based report.
