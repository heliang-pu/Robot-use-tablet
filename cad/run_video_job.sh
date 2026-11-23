#!/usr/bin/env bash
# Resume the gpt-6.2-astra Blender session with 视频任务.md; detached so a long render is not cut off.
cd "$(dirname "$0")"
codex exec resume 01a206c9-f14d-7f83-b91b-f4cb02fd5bd3 -m gpt-6.2-astra -c 'sandbox_mode="workspace-write"' \
  --skip-git-repo-check -o codex_video_last_message.txt - < 视频任务.md > codex_video_run.log 2>&1
echo "exit $?" > codex_video_exit.txt
