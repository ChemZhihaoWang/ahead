import os
import shutil
import tarfile
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_directory(directory_name: str, source_dir: str, output_dir: str) -> None:
    """Prepare and package a directory"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Directory path
    dir_path = os.path.join(source_dir, directory_name)
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory {directory_name} does not exist in {source_dir}")
    
    # Create tar file
    tar_path = os.path.join(output_dir, f"{directory_name}.tar")
    logging.info(f"Creating tar file: {tar_path}")
    
    with tarfile.open(tar_path, "w") as tar:
        tar.add(dir_path, arcname=directory_name)
    
    logging.info(f"Directory {directory_name} packaged successfully!")

def prepare_csv_files(source_dir: str, output_dir: str) -> None:
    """Package all CSV files together"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directory
    temp_dir = os.path.join(output_dir, "temp_label")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Copy all CSV files to temporary directory
    for file in os.listdir(source_dir):
        if file.endswith('.csv'):
            shutil.copy2(
                os.path.join(source_dir, file),
                os.path.join(temp_dir, file)
            )
            logging.info(f"Copied CSV file: {file}")
    
    # Create tar file
    tar_path = os.path.join(output_dir, "label.tar")
    logging.info(f"Creating CSV files tar package: {tar_path}")
    
    with tarfile.open(tar_path, "w") as tar:
        tar.add(temp_dir, arcname="label")
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    logging.info("CSV files packaged successfully!")

def main():
    # Source and output directories
    source_dir = r"D:\Desktop\hydro_channel\dataset"
    output_dir = r"D:\Desktop\hydro_channel\dataset\data_archives"
    
    # Process all subdirectories
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path) and not item.startswith('__'):
            try:
                prepare_directory(item, source_dir, output_dir)
            except Exception as e:
                logging.error(f"Error processing directory {item}: {str(e)}")
    
    # Process all CSV files
    try:
        prepare_csv_files(source_dir, output_dir)
    except Exception as e:
        logging.error(f"Error processing CSV files: {str(e)}")

if __name__ == "__main__":
    main() 