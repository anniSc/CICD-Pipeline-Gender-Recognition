import os

# The number of workflow files to create
num_files = 100

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

    os.rmdir(os.path.dirname(filename))

