python_files=("model/model_script/model_train.py" "deploy/deploy.py" "data/dataprep_scripts/datapreparation.py", "test/model_test_scripts/model_test.py")

# Run radon cc
for python_file in "${python_files[@]}"; do
    datetime=$(date +%Y%m%d_%H%M%S)
    output_file_cc="radon_tests/output_cc_max.txt"
    radon cc ${python_file} > ${output_file_cc}
done

# Run radon raw
for python_file in "${python_files[@]}"; do
    datetime=$(date +%Y%m%d_%H%M%S)
    output_file_raw="radon_tests/output_raw_max.txt"
    radon raw ${python_file} > ${output_file_raw}
done

# Run radon mi
for python_file in "${python_files[@]}"; do
    datetime=$(date +%Y%m%d_%H%M%S)
    output_file_mi="radon_tests/output_mi_max.txt"
    radon mi ${python_file} > ${output_file_mi}
done