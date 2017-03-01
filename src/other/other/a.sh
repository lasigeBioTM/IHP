python src/main.py load_corpus --goldstd hpo_train --log DEBUG
python src/main.py load_corpus --goldstd hpo_test --log DEBUG
python src/main.py train --goldstd hpo_train --models models/hpo_train --log DEBUG --entitytype hpo --crf crfsuite
python src/main.py test --goldstd hpo_test -o pickle data/results_hpo_train --models models/hpo_train --log DEBUG --entitytype hpo --crf crfsuite
python src/evaluate.py evaluate hpo_test --results data/results_hpo_train --models models/hpo_train --log DEBUG --entitytype hpo --rules andor stopwords small_ent longterms gowords posgowords
