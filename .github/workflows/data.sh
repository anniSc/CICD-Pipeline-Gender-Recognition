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

create_single_report() {
    local report_file=$1
    local report_title=$2
    local image_file=$3

    echo "## $report_title" > $report_file
    cml-publish "$image_file" --md >> $report_file
    cml-send-comment $report_file
}


for i in ${!distribution_files[@]}; do
  update_report ${distribution_files[$i]} ${distribution_names[$i]}
done

for file in data/plot_data/*.png; do
  echo "\n## Datenvisualisierung für $(basename "$file" .png)" >> $report_file
  cml-publish "$file" --md >> $report_file
done

echo "\n## Balancierte Daten Geschlechter" >> $report_file
cml-publish data/plots_balanced/Gender_balanced.png --md >> $report_file
echo "\n## Balancierte Daten Jung und Alt" >> $report_file
cml-publish data/plots_balanced/Young_balanced.png --md >> $report_file

cml-send-comment $report_file
