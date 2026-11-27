#!/usr/bin/env python3
"""Puts the mp3 files of a folder into the cockpit mock as playable tracks.

    python3 import_music.py /path/to/folder

Each song is copied to assets/www/music/NN.mp3, its embedded cover (if any) to NN.jpg, and
assets/www/music/list.js is rewritten with title, artist, album, length and the timed lyrics of a
.lrc file with the same name, if there is one. Title and artist come
from the tags, or from a file name like "title - artist.mp3". Needs ffmpeg and ffprobe.
Run ./build.sh install afterwards. The music folder is kept out of git.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "assets" / "www" / "music"


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:format_tags=title,artist,album",
         "-show_entries", "stream=codec_type,codec_name", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def lyrics(path):
    """Lines of a .lrc file as [seconds, text], sorted by time. Tags such as [ti:] are skipped."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("gb18030", errors="replace")
    offset = 0.0
    found = re.search(r"\[offset:\s*([+-]?\d+)\]", text)
    if found:
        offset = int(found.group(1)) / 1000.0
    out = []
    for line in text.splitlines():
        stamps = re.findall(r"\[(\d+):(\d+(?:\.\d+)?)\]", line)
        words = re.sub(r"<\d+:\d+(?:\.\d+)?>", "", re.sub(r"\[[^\]]*\]", "", line)).strip()
        if not stamps or not words:
            continue
        for minutes, seconds in stamps:
            out.append([round(max(0.0, int(minutes) * 60 + float(seconds) - offset), 2), words])
    return sorted(out)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    songs = sorted(src.glob("*.mp3"))
    if not songs:
        sys.exit(f"no mp3 files in {src}")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    tracks = []
    for n, song in enumerate(songs, 1):
        info = probe(song)
        tags = {k.lower(): v for k, v in info["format"].get("tags", {}).items()}
        name_title, _, name_artist = song.stem.partition(" - ")
        base = f"{n:02d}"
        shutil.copyfile(song, OUT / f"{base}.mp3")
        has_cover = any(s["codec_type"] == "video" for s in info["streams"])
        if has_cover:
            done = subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(song), "-an", "-frames:v", "1",
                 "-vf", "scale=400:400:force_original_aspect_ratio=increase,crop=400:400", str(OUT / f"{base}.jpg")])
            has_cover = done.returncode == 0
        tracks.append({
            "f": base,
            "t": tags.get("title") or name_title,
            "a": tags.get("artist") or name_artist or "未知歌手",
            "al": tags.get("album", ""),
            "d": round(float(info["format"]["duration"])),
            "cover": has_cover,
            "lrc": lyrics(song.with_suffix(".lrc")),
        })
        print(base, tracks[-1]["t"], "-", tracks[-1]["a"], tracks[-1]["d"], "s",
              "cover" if has_cover else "no cover", len(tracks[-1]["lrc"]), "lyric lines")
    (OUT / "list.js").write_text("'use strict';\nconst MUSIC = " + json.dumps(tracks, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"{len(tracks)} tracks written to {OUT}")


if __name__ == "__main__":
    main()
