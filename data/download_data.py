import os
import gzip
import shutil
import argparse
import requests

def download(url: str, path: str, force_download:bool=False):
    print(f"Downloading from {url}...")
    if os.path.exists(path) and not force_download:
        print(f"{path} already exists. Skipping download.")
        return 
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {response.status_code}")
    with open(path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded to {path}")

def extract(extract_from_path: str, extract_to_path: str, force_extract:bool=False):
    print("Extracting...")
    if os.path.exists(extract_to_path) and not force_extract:
        print(f"{extract_to_path} already exists. Skipping extraction.")
        return 
    with gzip.open(extract_from_path, 'rb') as f_in, open(extract_to_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"Extracted to {extract_to_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract a .gz file.")
    parser.add_argument('--url', type=str, required=True, help='URL of the .gz file')
    parser.add_argument('--output', type=str, default='data/finefoods.txt.gz', help='Output path for the .gz file')
    parser.add_argument('--force', action='store_true', help='Force re-download and extraction')

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    txt_path = args.output.replace('.gz', '')
    download(args.url, args.output, force_download=args.force)
    extract(args.output, txt_path, force_extract=args.force)