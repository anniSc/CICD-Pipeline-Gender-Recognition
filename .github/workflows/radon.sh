#!/bin/bash

run_radon() {
    local python_file=$1

    datetime=$(date +%Y%m%d_%H%M%S)
    output_file="output_${datetime}_${python_file}.txt"

    radon cc ${python_file} > ${output_file}
    radon raw ${python_file} >> ${output_file}
    radon mi ${python_file} >> ${output_file}
}

# List of python files
python_files=("model/model_script/model_train.py" "deploy/deploy.py" "data/dataprep_scripts/datapreparation.py", "test/model_test_scripts/model_test.py", "test/data_test_scripts/data_test.py")

for python_file in "${python_files[@]}"; do
    run_radon $python_file
done