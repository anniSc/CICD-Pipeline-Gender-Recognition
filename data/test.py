import glob
import shutil
import os

source_dir = 'C:/Users/busse/CICDPipeline/BA-CICD-Pipeline/data/img_align_celeba'  # Source directory
dest_dir = 'C:/Users/busse/BA/BA-CICD-Pipeline/data/img_align_celeba'  # Destination directory

# Create destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Get all .jpg files in the source directory
files = glob.glob(os.path.join(source_dir, '*.jpg'))

for file in files:
    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(file))[0]
    
    # Check if the filename is a number between 17004 and 30000
    if filename.isdigit() and 200600 <= int(filename) <= 202599:
        # Copy the file to the destination directory
        shutil.copy(file, dest_dir)