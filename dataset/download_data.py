import argparse
import os
import logging
import tarfile
import requests
import shutil
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Google Drive configuration
GOOGLE_DRIVE_CONFIG = {
    "file_ids": {
        "sampling_200": "14torwTRqZyfHjJxxi-KF_MlljPWoB3g5",
        "sampling_400": "1VT3qSaYJ1e37wjCNmGKEAX_quOVrcKdR",
        "sampling_600": "11nkDF3w3EPr8kCcCDd_yqZ0D2TKtnO1b",
        "sampling_800": "15LxWGrn1FEZJklAMmx7Gr8onHScgxfx8",
        "hydrofig": "1NanEfr0znX133owEkAzKvn9wmY_LvtRN",
        "SISSO": "18y7Ja3RwXjyfLZaBFPg7xOrKypr8uDTO",
        "test_images": "1hLXuxq7uFXe9Ly7y1lanYWaMqJpedjww",
        "label": "1FO5F20xZdC9MysnDQmvdofx-GUZhiHek",
        "correlation": "1ll8DtXIIwrtlW6CnJ67vC7N6ouOUqPv_",
        "randomfig_100w": "1kXxXIs_uT4msnK317y8V1clvTv-2oMAd"
    }
}

def get_google_drive_download_url(file_id: str) -> str:
    """Generate Google Drive download URL"""
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def download_file(url: str, save_path: str, max_retries: int = 3) -> None:
    """Download file with progress bar and retry mechanism"""
    for attempt in range(max_retries):
        try:
            # Create a temporary file
            temp_path = save_path + '.temp'
            
            # Download with requests
            session = requests.Session()
            response = session.get(url, stream=True)
            
            # Handle Google Drive virus scan warning
            if 'google.com' in url:
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        confirm = value
                        url = f"{url}&confirm={confirm}"
                        response = session.get(url, stream=True)
                        break
            
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(temp_path, 'wb') as f, tqdm(
                desc=os.path.basename(save_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)
            
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                os.rename(temp_path, save_path)
                return
            else:
                logging.warning(f"Download attempt {attempt + 1} failed: File verification failed")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
        except Exception as e:
            logging.error(f"Download attempt {attempt + 1} failed: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
    raise Exception("Failed to download file after multiple attempts")

def extract_file(file_path: str, extract_path: str) -> None:
    """Extract tar file"""
    try:
        # Ensure extract path exists
        os.makedirs(extract_path, exist_ok=True)
        
        # Extract tar file
        with tarfile.open(file_path, 'r:*') as tar:
            tar.extractall(path=extract_path)
            logging.info(f"Successfully extracted {len(tar.getnames())} files from tar archive")
            
    except Exception as e:
        logging.error(f"Extraction failed: {str(e)}")
        raise

def get_data(datadir: str, dataset_name: str, keep_archive: bool = False) -> None:
    """Download and process dataset"""
    os.makedirs(datadir, exist_ok=True)
    
    if dataset_name not in GOOGLE_DRIVE_CONFIG['file_ids']:
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(GOOGLE_DRIVE_CONFIG['file_ids'].keys())}")
    
    file_id = GOOGLE_DRIVE_CONFIG['file_ids'][dataset_name]
    download_link = get_google_drive_download_url(file_id)
    
    archive_path = os.path.join(datadir, f"{dataset_name}.tar")
    extract_path = datadir  
    
    # Download file
    logging.info(f"Downloading dataset {dataset_name}...")
    download_file(download_link, archive_path)
    
    # Extract file
    logging.info(f"Extracting dataset {dataset_name}...")
    extract_file(archive_path, extract_path)
    
    # Clean up
    if not keep_archive:
        os.remove(archive_path)
        logging.info(f"Removed temporary file {archive_path}")

if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="Download datasets from Google Drive")
    parser.add_argument("--dataset", type=str, required=True, 
                       help="Dataset to download")
    parser.add_argument("--data-dir", type=str, default=current_dir,
                       help="Directory to save dataset")
    parser.add_argument("--keep-archive", action="store_true",
                       help="Keep downloaded archive file")
    parser.add_argument("--retries", type=int, default=3,
                       help="Number of download retries")
    
    args = parser.parse_args()
    
    try:
        get_data(args.data_dir, args.dataset, args.keep_archive)
        logging.info(f"Dataset {args.dataset} downloaded successfully!")
    except Exception as e:
        logging.error(f"Download failed: {str(e)}") 