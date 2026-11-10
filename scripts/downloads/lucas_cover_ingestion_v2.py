import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Target URL and local destination
BASE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/LUCAS/LUCAS_COVER/LUCAS2018/"
DOWNLOAD_DIR = "./data/raw/LUCAS/LUCAS_COVER/LUCAS2018_data"

def download_recursive(url, current_local_dir):
    # Ensure local directory exists
    os.makedirs(current_local_dir, exist_ok=True)
    
    try:
        response = requests.get(url, timeout=(10,120))
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
            file_name = href
            local_file_path = os.path.join(current_local_dir, file_name)
            temp_file_path = local_file_path + ".tmp"
            
            # Skip if the file already exists completely
            if os.path.exists(local_file_path):
                print(f"Skipping {file_name} (already exists)")
                continue
                
            print(f"Downloading: {file_name} ...", end="", flush=True)
            try:
                with requests.get(full_url, stream=True, timeout=(10,120)) as file_res:
                    file_res.raise_for_status()
                    
                    total_size = int(file_res.headers.get('content-length', 0))
                    
                    # Download to a temporary file first
                    with open(temp_file_path, 'wb') as f:
                        for chunk in file_res.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                
                    # Verify size if content-length is provided by server
                    if total_size > 0 and os.path.getsize(temp_file_path) != total_size:
                        print(f" Failed! (Incomplete download)")
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                        continue
                        
                    # Move temp file to final destination once successful
                    os.replace(temp_file_path, local_file_path)
                    print(" Done!")
            except requests.RequestException as e:
                print(f" Failed! ({e})")
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

if __name__ == "__main__":
    print(f"Starting download from {BASE_URL}")
    download_recursive(BASE_URL, DOWNLOAD_DIR)
    print("\nAll downloads finished!")