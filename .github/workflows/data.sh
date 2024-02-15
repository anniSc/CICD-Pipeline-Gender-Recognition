#!/bin/bash

# Declare variables
report_file_distribution="reports/report_distribution.md"
report_file="reports/report_data_plots.md"
distribution_files=("data/report_data/exponential_distribution.txt" "data/report_data/binomia_distribution.txt" "data/report_data/norm_distribution.txt" "data/report_data/uniform_distribution.txt")
distribution_names=("Exponentialverteilung" "Binomialverteilung" "Normalverteilung" "Uniformverteilung")

# Function to check file and update report
update_report() {
  if [ -s "$1" ]
  then
    echo "## Daten folgen wahrscheinlich einer $2!" > $report_file_distribution
    cat $1 >> $report_file_distribution
  else
    echo " "
  fi
}

# Function to create a single report
create_single_report() {
    local report_file=$1
    local report_title=$2
    local image_file=$3

    echo "## $report_title" > $report_file
    cml-publish "$image_file" --md >> $report_file
    cml-send-comment $report_file
}

# Function to update report with data visualization
update_report_with_visualization() {
    local report_file=$1
    local file_directory=$2

    for file in $file_directory; do
        echo "\n## Datenvisualisierung für $("$file" .png)" >> $report_file
        cml-publish "$file" --md >> $report_file
    done
}

# # Update report with distribution data
# for i in ${!distribution_files[@]}; do
#   update_report ${distribution_files[$i]} ${distribution_names[$i]}
# done

# Update report with data visualization
update_report_with_visualization $report_file "data/plot_data/*.png"

# Create single reports
create_single_report $report_file "Balancierte Daten Geschlechter" "data/plots_balanced/Gender_balanced.png"
create_single_report $report_file "Balancierte Daten Jung und Alt" "data/plots_balanced/Young_balanced.png"
create_single_report $report_file "Balancierte Daten BMI" "data/plots_balanced/BMI_balanced.png"
cml-send-comment $report_file