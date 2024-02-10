import os

# The number of workflow files to create
num_files = 10

for i in range(1, num_files + 1):
    # The filename
    filename = f".github/workflows/workflow{i}.yml"

    # The content of the file
    content = f"""
name: Workflow {i}
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: Run a one-line script
      run: echo Hello, world!
"""

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Write the content to the file
    with open(filename, "w") as f:
        f.write(content)