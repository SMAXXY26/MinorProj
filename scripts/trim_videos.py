"""
trim_videos.py  —  Split long videos into 60-second chunks
Usage:
    python scripts/trim_videos.py
    python scripts/trim_videos.py --input data/sequences/negative_clips
    python scripts/trim_videos.py --input data/sequences/negative_clips --output data/sequences/chunks
"""

import os
import subprocess
import argparse
from pathlib import Path


def trim_video(input_path, output_dir, chunk_duration=60):
    """Split a video into fixed-length chunks using ffmpeg."""
    stem   = Path(input_path).stem
    suffix = Path(input_path).suffix

    # Get video duration
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path,
    ], capture_output=True, text=True)

    try:
        duration = float(result.stdout.strip())
    except Exception:
        print(f"  [SKIP] Could not read duration: {input_path}")
        return 0

    n_chunks = max(1, int(duration // chunk_duration))
    print(f"  {Path(input_path).name}  ({duration:.0f}s → {n_chunks} chunks)")

    created = 0
    for i in range(n_chunks):
        start    = i * chunk_duration
        out_path = os.path.join(output_dir, f"{stem}_chunk{i:03d}{suffix}")
        result   = subprocess.run([
            "ffmpeg", "-y",
            "-ss",  str(start),
            "-i",   input_path,
            "-t",   str(chunk_duration),
            "-c",   "copy",
            "-loglevel", "error",
            out_path,
        ])
        if result.returncode == 0 and Path(out_path).exists():
            created += 1

    return created


if __name__ == "__main__":
    # Resolve project root — one level up from scripts/
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    type=str,
                        default=str(project_root / "data/sequences/negative_clips"))
    parser.add_argument("--output",   type=str,  default=None,
                        help="Output folder (default: same as input)")
    parser.add_argument("--duration", type=int,  default=60)
    args = parser.parse_args()

    # ── Resolve output dir BEFORE using it ───────────────────────────────
    out_dir = Path(args.output or args.input)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Install ffmpeg if needed
    os.system("which ffmpeg > /dev/null || sudo apt-get install -y ffmpeg")

    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos     = [
        str(p) for p in Path(args.input).iterdir()
        if p.suffix.lower() in video_exts
        and "chunk" not in p.stem
    ]

    if len(videos) == 0:
        print(f"No videos found in {args.input}")
        exit(0)

    print(f"Found {len(videos)} videos to trim")
    print(f"Output dir : {out_dir}\n")

    deleted = 0
    for v in videos:
        chunks_created = trim_video(v, str(out_dir), args.duration)

        if chunks_created > 0:
            Path(v).unlink()
            deleted += 1
            print(f"  [Deleted] {Path(v).name}  ({chunks_created} chunks created)")
        else:
            print(f"  [Kept]    {Path(v).name} — no chunks created")

    print(f"\nDone!")
    print(f"  Videos processed : {len(videos)}")
    print(f"  Originals deleted: {deleted}")
    print(f"  Chunks saved to  : {out_dir}")