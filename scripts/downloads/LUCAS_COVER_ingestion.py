import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Target URL and local destination
BASE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/LUCAS/LUCAS_COVER/LUCAS2018/"
DOWNLOAD_DIR = "LUCAS2018_data"

def download_recursive(url, current_local_dir):
    # Ensure local directory exists
    os.makedirs(current_local_dir, exist_ok=True)
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to access {url}: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Iterate through all links in the directory listing
    for link in soup.find_all('a'):
        href = link.get('href')
        
        # Skip parent directory links, absolute external links, or empty links
        if not href or href.startswith('?') or href.startswith('/') or '..' in href:
            continue
            
        # Build full absolute URL
        full_url = urljoin(url, href)
        
        # Check if the link is a sub-directory (ends with '/')
        if href.endswith('/'):
            sub_dir_name = href.strip('/')
            next_local_dir = os.path.join(current_local_dir, sub_dir_name)
            print(f"\nEntering directory: {sub_dir_name}")
            download_recursive(full_url, next_local_dir)
        else:
            # It's a file, download it
            file_name = href
            local_file_path = os.path.join(current_local_dir, file_name)
            
            # Optional: Skip download if file already exists to allow resuming
            if os.path.exists(local_file_path):
                print(f"Skipping {file_name} (already exists)")
                continue
                
            print(f"Downloading: {file_name} ...", end="", flush=True)
            try:
                with requests.get(full_url, stream=True, timeout=30) as file_res:
                    file_res.raise_for_status()
                    with open(local_file_path, 'wb') as f:
                        for chunk in file_res.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(" Done!")
            except requests.RequestException as e:
                print(f" Failed! ({e})")

if __name__ == "__main__":
    print(f"Starting download from {BASE_URL}")
    download_recursive(BASE_URL, DOWNLOAD_DIR)
    print("\nAll downloads finished!")
