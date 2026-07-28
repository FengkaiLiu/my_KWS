import time
import tarfile
import zipfile
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

# We disable TLS verification below (verify=False) because a SOCKS proxy in the
# path can present a mismatched cert for download.tensorflow.org. The source is
# a known-good public dataset, so this is an acceptable trade-off here. Silence
# the resulting per-request warning so it doesn't drown the progress bar.
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


DATA_RAW = Path("data/raw")


def _download_with_resume(url: str, tmp_path: Path, max_retries: int = 10,
                          timeout: int = 30, chunk_size: int = 1 << 16):
    """
    Download `url` to `tmp_path` with HTTP-Range resume.

    If the connection drops, rerun the script: it picks up from the bytes
    already on disk instead of starting over. Retries a stalled connection
    up to max_retries times with a short backoff.
    """
    for attempt in range(1, max_retries + 1):
        existing = tmp_path.stat().st_size if tmp_path.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}

        try:
            with requests.get(url, stream=True, headers=headers,
                              timeout=timeout, verify=False) as r:
                # 206 = partial (resume accepted); 200 = full (server ignored Range)
                if existing and r.status_code == 200:
                    existing = 0
                    tmp_path.unlink(missing_ok=True)
                elif r.status_code not in (200, 206):
                    r.raise_for_status()

                content_len = int(r.headers.get("content-length", 0))
                total = existing + content_len if content_len else None

                mode = "ab" if existing else "wb"
                with open(tmp_path, mode) as f, tqdm(
                    total=total, initial=existing, unit="B", unit_scale=True,
                    desc=f"{tmp_path.name} (try {attempt})",
                ) as bar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            return
        except (requests.exceptions.RequestException, OSError) as e:
            have = tmp_path.stat().st_size if tmp_path.exists() else 0
            wait = min(5 * attempt, 30)
            print(f"\n[!] interrupted ({type(e).__name__}: {e}). "
                  f"Retry {attempt}/{max_retries} in {wait}s "
                  f"(resuming from {have} bytes)...")
            time.sleep(wait)

    raise RuntimeError(f"Failed to download {url} after {max_retries} retries. "
                       f"Partial file kept at {tmp_path} - just rerun to resume.")


def download_and_extract(url: str, dest_dir: Path, extracted_name: str | None = None):
    dest_dir = Path(dest_dir)
    if dest_dir.exists():
        print(f"Already exist, skip: {dest_dir}")
        return

    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_dir.parent / Path(url).name

    _download_with_resume(url, tmp_path)

    print(f"Extracting {tmp_path.name} ...")
    if url.endswith(".tar.gz"):
        dest_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tmp_path) as tf:
            tf.extractall(dest_dir)
    else:
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(dest_dir.parent)
        shutil.move(str(dest_dir.parent / extracted_name), str(dest_dir))

    tmp_path.unlink()
    print(f"Done: {dest_dir}")


if __name__ == "__main__":
    download_and_extract(
        "https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        DATA_RAW / "speech_commands",
    )
    download_and_extract(
        "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip",
        DATA_RAW / "esc50",
        "ESC-50-master",
    )