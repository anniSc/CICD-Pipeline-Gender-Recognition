# Dieses Skript berechnet die Radon-Metriken für den Quellcode und speichert sie in separaten Dateien.
# Es wird im Job "code_quality_check" verwendet.

# Berechnet den Bericht für die zyklomatische Komplexität.
# Die zyklomatische Komplexität misst die Anzahl der linear unabhängigen Pfade durch den Code.
radon cc model/model_script/model_train.py > radon_tests/radon_cc_model_train.txt
radon cc deploy/deploy.py > radon_tests/radon_cc_deploy.txt
radon cc data/dataprep_scripts/datapreparation.py > radon_tests/radon_cc_datapreparation.txt 
radon cc test/model_test_scripts/model_test.py > radon_tests/radon_cc_model_test.txt

# Berechnet den Bericht mit dem Wartbarkeitsindex für die statische Codeanalyse.
# Der Wartbarkeitsindex misst, wie einfach der Code zu warten ist. Die Skala reicht von -100 bis 100.
radon mi deploy/deploy.py > radon_tests/radon_mi_deploy.txt 
radon mi model/model_script/model_train.py > radon_tests/radon_mi_model_train.txt
radon mi data/dataprep_scripts/datapreparation.py > radon_tests/radon_mi_datapreparation.txt
radon mi test/model_test_scripts/model_test.py > radon_tests/radon_mi_model_test.txt

# Berechnet die Rohmetriken, die eine einfache und hochrangige Beschreibung des Quellcodes liefern.
# Die Rohmetriken umfassen die Anzahl der logischen, leeren, Code- und Kommentarzeilen.
radon raw model/model_script/model_train.py > radon_tests/radon_raw_model_train.txt
radon raw deploy/deploy.py > radon_tests/radon_raw_deploy.txt 
radon raw data/dataprep_scripts/datapreparation.py > radon_tests/radon_raw_datapreparation.txtradon raw test/model_test_scripts/model_test.py > radon_tests/radon_raw_model_test.txt

