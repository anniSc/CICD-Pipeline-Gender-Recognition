import os
import shutil

def copy_python_scripts(source_directory, destination_directory):
    """Copy all Python scripts from the source directory to the destination directory."""
    # List all files in the source directory
    files = os.listdir(source_directory)

    # Filter the list to include only Python scripts
    python_scripts = [file for file in files if file.endswith('.py')]

    # Copy each Python script to the destination directory
    for script in python_scripts:
        source = os.path.join(source_directory, script)
        destination = os.path.join(destination_directory, script)
        shutil.copy(source, destination)



copy_python_scripts('deploy', 'unittest')
copy_python_scripts('data/dataprep_scripts/', 'unittest')