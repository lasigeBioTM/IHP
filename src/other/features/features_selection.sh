python src/main.py train --goldstd hpo_train --models models/hpo_train
python src/main.py test --goldstd hpo_test -o pickle data/results_hpo_train --models models/hpo_train
python src/evaluate.py evaluate hpo_test --results data/results_hpo_train --models models/hpo_train
