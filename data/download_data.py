import os
import gzip
import shutil
import argparse
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def download(url: str, path: str, force_download: bool = False):
    logger.info(f"Downloading from {url}...")
    if os.path.exists(path) and not force_download:
        logger.info(f"{path} already exists. Skipping download.")
        return
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        logger.error(f"Failed to download file: {response.status_code}")
        raise Exception(f"Download failed with status code {response.status_code}")
    with open(path, 'wb') as f:
        f.write(response.content)
    logger.info(f"Downloaded to {path}")

def extract(src_path: str, dst_path: str, force_extract: bool = False):
    logger.info("Extracting...")
    if os.path.exists(dst_path) and not force_extract:
        logger.info(f"{dst_path} already exists. Skipping extraction.")
        return
    with gzip.open(src_path, 'rb') as f_in, open(dst_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    logger.info(f"Extracted to {dst_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract a .gz file.")
    parser.add_argument('--url', type=str, required=True, help='URL of the .gz file')
    parser.add_argument('--output', type=str, default='data/raw/finefoods.txt.gz', help='Output path for the .gz file')
    parser.add_argument('--force', action='store_true', help='Force re-download and extraction')

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    txt_path = args.output.replace('.gz', '')
    download(args.url, args.output, force_download=args.force)
    extract(args.output, txt_path, force_extract=args.force)