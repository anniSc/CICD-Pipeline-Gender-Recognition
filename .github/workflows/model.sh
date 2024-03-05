# Funktion um den Report zu aktualisieren, zu senden und zu erstellen
create_report(){
    local report_file=$1
    local report_title=$2
    local image_dir=$3

    echo "## $report_title" > $report_file
    for file in $image_dir/*.png; do
        echo "Publishing $file"
        cml-publish "$file" --md >> $report_file
    done
    cml-send-comment $report_file
}


# Funktion um einzelnen Report zu erstellen, zu aktualisieren und zu senden
create_single_report(){
    local report_file=$1
    local report_title=$2
    local image_file=$3

    echo "## $report_title" > $report_file
    cml-publish "$image_file" --md >> $report_file
    cml-send-comment $report_file
}

# Funktion um den spezifischen Report zu aktualisieren, zu senden und zu erstellen
create_ml_report(){
    echo "## Model Metriken" > report_ml.md
    cat test/metrics/metrics.txt >> report_ml.md
    echo "\n## Fairlearn Ergebnisse" >> report_ml.md
    cml-publish test/metrics/metrics.jpg --md >> report_ml.md
    cml-publish test/metricsFairlearn/Fig1metricsFairLearn.jpg --md >> report_ml.md
    cml-publish test/metricsFairlearn/Fig2metricsFairLearn.jpg --md >> report_ml.md
    cml-send-comment report_ml.md
}

create_ml_report
# create_report "report_test_rauschen.md" "Modellmetriken mit verauschten Bilder" "test/test-plots-rauschen"
# create_report "report_test_verzerrung.md" "Modellmetriken mit verauschten Bilder" "test/test-plots-verzerrung"
# create_report "report_rotation.md" "Modellmetriken mit verdrehte Bilder" "test/test-plots-rotation"
# create_report "reports/report_activation.md" "Erklärbarkeit des Modells" "test/activation_map"
create_report "reports/report_test_rauschen.md" "Modellmetriken mit verauschten Bilder" "test/test-plots-rauschen"
create_report "reports/report_test_verzerrung.md" "Modellmetriken mit verauschten Bilder" "test/test-plots-verzerrung"
create_report "reports/report_rotation.md" "Modellmetriken mit verdrehte Bilder" "test/test-plots-rotation"
create_report "reports/report_activation.md" "Erklärbarkeit des Modells" "test/activation_map"