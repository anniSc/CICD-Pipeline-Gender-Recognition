# Funktion, um einen einzelnen Bericht zu erstellen.
# Parameter:
#   - report_file: Der Dateiname des Berichts.
#   - report_title: Der Titel des Berichts.
#   - image_file: Der Dateiname des Bildes, das dem Bericht hinzugefügt werden soll.
create_single_report() {
    local report_file=$1
    local report_title=$2
    local image_file=$3

    echo "## $report_title" > $report_file
    cml-publish "$image_file" --md >> $report_file
    cml-send-comment $report_file
}


create_single_report "report_ml.md" "CPU/Speicherauslastung" "model/cpu_memory_usage.png"