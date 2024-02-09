           report_file_distribution="report_distribution.md"
            expoFile="data/report_data/exponential_distribution.txt"
            if [ -s "$file" ]
            then
              echo "## Daten folgen wahrscheinlich einer Exponentialverteilung!" > $report_file_distribution
              cat $expoFile >> $report_file_distribution
            else
              echo " "
            fi
            binomiaFile="data/report_data/binomia_distribution.txt"
            if [ -s "$file" ]
            then
              echo "## Daten folgen wahrscheinlich einer Binomialverteilung!" > $report_file_distribution
              cat $binomiaFile >> $report_file_distribution
            else
              echo " "
            fi
            normFile="data/report_data/norm_distribution.txt"
            if [ -s "$file" ]
            then
              echo "## Daten folgen wahrscheinlich einer Normalverteilung!" > $report_file_distribution
              cat $normFile >> $report_file_distribution
            else
              echo " "
            fi
            uniformFile="data/report_data/uniform_distribution.txt"
            if [ -s "$file" ]
            then
              echo "## Daten folgen wahrscheinlich einer Uniformverteilung!" > $report_file_distribution
              cat $uniformFile >> $report_file_distribution
            else
              echo " "
            fi
            report_file="report_data_plots.md"
            # Iterate over each file in the plot_data directory
            for file in data/plot_data/*.png; do
              # Update the report
              echo "\n## Datenvisualisierung für $(basename "$file" .png)" >> $report_file
              cml-publish "$file" --md >> $report_file
            done
            echo "\n## Balancierte Daten Geschlechter" >> $report_file
            cml-publish data/plots_balanced/Gender_balanced.png
            echo "\n## Balancierte Daten Jung und Alt" >> $report_file
            cml-publish data/plots_balanced/Young_balanced.png

            cml-send-comment $report_file