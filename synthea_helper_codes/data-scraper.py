import os
import glob
import random
import shutil
from pathlib import Path

# Set this to the root folder of your Synthea dataset
# The script will search all subfolders inside this path.
SOURCE_DATA_DIR = r"data/synthea_dataset/fhir"

# Set this to the new folder where you want to copy the 200 files
# This folder will be created if it doesn't exist.
EVAL_DIR = r"data/evaluation_set_2000"

NUM_FILES_TO_SELECT = 2000

# --- 2. MAIN SCRIPT ---

def select_and_copy_files():
    print(f"Scanning for all .json files in: {SOURCE_DATA_DIR}")
    
    # Use Pathlib's glob to recursively find all .json files
    source_path = Path(SOURCE_DATA_DIR)
    all_files = list(source_path.glob('**/*.json'))
    
    print(f"Found {len(all_files)} total files.")

    # Check if we have enough files 
    if len(all_files) < NUM_FILES_TO_SELECT:
        print(f"Error: You asked for {NUM_FILES_TO_SELECT} files, but only {len(all_files)} were found.")
        print("Please check your SOURCE_DATA_DIR path.")
        return

    # Create the destination folder 
    try:
        os.makedirs(EVAL_DIR, exist_ok=True)
        print(f"Created or found destination folder: {EVAL_DIR}")
    except Exception as e:
        print(f"Error creating directory: {e}")
        return

    # Randomly select 2000 files 
    random.seed(42) # Use a fixed seed for a reproducible random choice
    selected_files = random.sample(all_files, NUM_FILES_TO_SELECT)
    
    print(f"\nRandomly selected {len(selected_files)} files. Starting copy...")

    # Copy the files 
    for i, file_path in enumerate(selected_files):
        try:
            # shutil.copy will copy the file to the new directory
            shutil.copy(file_path, EVAL_DIR)
            if (i + 1) % 20 == 0:
                print(f"  ...copied {i + 1} / {NUM_FILES_TO_SELECT} files.")
        except Exception as e:
            print(f"Error copying {file_path}: {e}")

    print("\n--- Process Complete! ---")
    print(f"Successfully copied {NUM_FILES_TO_SELECT} random files to {EVAL_DIR}")
    print("You can now zip this folder and share it with your team for labeling.")

if __name__ == "__main__":
    select_and_copy_files()