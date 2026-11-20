[README.md](https://github.com/user-attachments/files/35127368/README.md)
# Auto Clipper

discover.py  -> finds long videos / finished streams (search queries + channels in config.json)
clipper.py   -> finds best moments (audio, transcript, most-replayed, live-chat spikes) and cuts 9:16 clips
pipeline.py  -> runs both, remembers processed videos (processed.json), builds output/index.json review queue

## Try locally (needs ffmpeg)
    pip install -r requirements.txt
    python discover.py        # preview what it would pick, no download
    python pipeline.py        # full run

## GitHub
Upload everything; put clip.yml at .github/workflows/clip.yml.
Runs every 12h or manually (Actions -> Auto clip pipeline -> Run workflow).
Clips come as a downloadable artifact. Edit config.json to change topics/channels.
If YouTube blocks downloads, add cookies.txt content as secret YT_COOKIES.

Only clip content you own or have permission to use; credit the creator.
