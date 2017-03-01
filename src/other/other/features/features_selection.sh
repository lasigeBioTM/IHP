python src/main.py train --goldstd hpo_train --models models/hpo_train
python src/main.py test --goldstd hpo_test -o pickle data/results_hpo_train --models models/hpo_train
python src/evaluate.py evaluate hpo_test --results data/results_hpo_train --models models/hpo_train --rules andor stopwords small_ent twice_validated stopwords gowords posgowords longterms small_len quotes defwords digits lastwords


python src/main.py train --goldstd hpo_train --models models/hpo_train; python src/main.py test --goldstd hpo_test -o pickle data/results_hpo_train --models models/hpo_train; python src/evaluate.py evaluate hpo_test --results data/results_hpo_train --models models/hpo_train