
run_radon_cc() {
    local python_file=$1
    output_file_cc="radon_tests/output_radon_cc_max.txt"
    radon cc ${python_file} > ${output_file_cc}
}

run_radon_raw() {
    local python_file=$1
    output_file_raw="radon_tests/output_radon_raw_max.txt"
    radon raw ${python_file} > ${output_file_raw}
}

run_radon_mi() {
    local python_file=$1
    output_file_mi="radon_tests/output_radon_mi_max.txt"
    radon mi ${python_file} > ${output_file_mi}
}

# List of python files
python_files=("model/model_script/model_train.py" "deploy/deploy.py" "data/dataprep_scripts/datapreparation.py", "test/model_test_scripts/model_test.py")

for python_file in "${python_files[@]}"; do
    run_radon_cc $python_file
    run_radon_raw $python_file
    run_radon_mi $python_file
done