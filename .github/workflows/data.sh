#!/bin/bash

# Declare variables
report_file_distribution="report_distribution.md"
report_file="report_data_plots.md"
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

# Check each distribution file and update report
for i in ${!distribution_files[@]}; do
  update_report ${distribution_files[$i]} ${distribution_names[$i]}
done

# Iterate over each file in the plot_data directory
for file in data/plot_data/*.png; do
  # Update the report
  echo "\n## Datenvisualisierung für $(basename "$file" .png)" >> $report_file
  cml-publish "$file" --md >> $report_file
done

# Update report with balanced data
echo "\n## Balancierte Daten Geschlechter" >> $report_file
cml-publish data/plots_balanced/Gender_balanced.png --md >> $report_file
echo "\n## Balancierte Daten Jung und Alt" >> $report_file
cml-publish data/plots_balanced/Young_balanced.png --md >> $report_file

# Send the report
cml-send-comment $report_file