import requests
import tarfile
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm


DATA_RAW = Path("data/raw")

def download_and_extract(url: str, dest_dir: Path, extracted_name: str | None = None):
    dest_dir = Path(dest_dir)
    if dest_dir.exists():
        print(f"Already exist, skip: {dest_dir}")
        return

    response = requests.get(url, stream=True)
    tmp_path = dest_dir.parent / Path(url).name
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    total = int(response.headers.get("content-length", 0))
    with open(tmp_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=tmp_path.name) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    if url.endswith(".tar.gz"):
        dest_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tmp_path) as tf:
            tf.extractall(dest_dir)
    else:
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(dest_dir.parent)
        shutil.move(str(dest_dir.parent / extracted_name), str(dest_dir))

    tmp_path.unlink()


if __name__ == "__main__":
    download_and_extract(
        "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        DATA_RAW / "speech_commands",
    )
    download_and_extract(
        "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip",
        DATA_RAW / "esc50",
        "ESC-50-master",
    )
