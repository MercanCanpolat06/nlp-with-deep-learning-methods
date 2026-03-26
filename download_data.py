#downloads the necessary data

import urllib.request
import zipfile
import os

def download_text8(save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)
    zip_path = os.path.join(save_dir, "text8.zip")
    txt_path = os.path.join(save_dir, "text8")
    
    if not os.path.exists(zip_path) and not os.path.exists(txt_path):
        print("downloading text8 dataset")
        url = "http://mattmahoney.net/dc/text8.zip"
        urllib.request.urlretrieve(url, zip_path)
        print("download complete!")
    if not os.path.exists(txt_path):
        print("Extracting the zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
        print(f"Data ready:: {txt_path}")
    else:
        print(f"Data already exists: {txt_path}")

if __name__ == "__main__":
    download_text8()