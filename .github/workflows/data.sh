#!/bin/bash

# Deklaration der Variablen
report_file_distribution="reports/report_distribution.md"
report_file="reports/report_data_plots.md"
distribution_files=("data/report_data/exponential_distribution.txt" "data/report_data/binomia_distribution.txt" "data/report_data/norm_distribution.txt" "data/report_data/uniform_distribution.txt")
distribution_names=("Exponentialverteilung" "Binomialverteilung" "Normalverteilung" "Uniformverteilung")

# Funktion um den Report zu aktualisieren, zu senden und zu erstellen
update_report() {
  if [ -s "$1" ]
  then
    echo "## Daten folgen wahrscheinlich einer $2!" > $report_file_distribution
    cat $1 >> $report_file_distribution
  else
    echo " "
  fi
}

# Funktion um den Report mit Visualisierungen zu aktualisieren, zu senden und zu erstellen
create_single_report() {
    local report_file=$1
    local report_title=$2
    local image_file=$3

    echo "## $report_title" > $report_file
    cml-publish "$image_file" --md >> $report_file
    cml-send-comment $report_file
}

# Funktion um den Report mit Visualisierungen zu aktualisieren, zu senden und zu erstellen
update_report_with_visualization() {
    local report_file=$1
    local file_directory=$2

    for file in "$file_directory"/*; do
        echo "Datenvisualisierung für $(basename "$file" .png)" >> $report_file
        cml-publish "$file" --md >> $report_file
    done
    cml-send-comment $report_file
}

# ausführen der Funktionen
update_report_with_visualization $report_file "data/plot_data/"
create_single_report $report_file "Balancierte Daten Geschlechter" "data/plots_balanced/balanced_gender.png"
create_single_report $report_file "Balancierte Daten Jung und Alt" "data/plots_balanced/balanced_young.png"

create_single_report $report_file_distribution "Verteilung der Daten" "data/reports_data/binomial_distribution.txt"
create_single_report $report_file_distribution "Verteilung der Daten" "data/reports_data/norm_distribution.txt"
create_single_report $report_file_distribution "Verteilung der Daten" "data/reports_data/exponential_distribution.txt"
create_single_report $report_file_distribution "Verteilung der Daten" "data/reports_data/uniform_distribution.txt"
cml-send-comment $report_file