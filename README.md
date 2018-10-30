# IHP
Framework for identifying Human Phenotype entities

Dependencies and other uses should follow the original ReadMe.

This is a fork created to accomodate an annotator for the [Human Phenotype Ontology](http://human-phenotype-ontology.github.io).
It uses Gold Standard Corpora and Test Suites Created by Bio-Lark. [Link Here](http://bio-lark.org/hpo_res.html)

# Usage
If a corpus is to be loaded into IHP, it's necessary to run Stanford CoreNLP. 
   ```
   cd bin/stanford-corenlp-full-2015-12-09/
   java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 500000 &
   ```

##Load Corpus (For both Gold Standard Corpora and Test Suite)
```
   python src/main.py load_corpus --goldstd hpo_train --log DEBUG
   python src/main.py load_corpus --goldstd hpo_test --log DEBUG
   python src/main.py load_corpus --goldstd tsuite --log DEBUG
```
   
##Train, Test and Evaluate with StanfordNER
```
   python src/main.py train --goldstd hpo_train --models models/hpo_train --log DEBUG
   python src/main.py test --goldstd hpo_test -o pickle data/results_hpo_train --models models/hpo_train --log DEBUG
   python src/evaluate.py evaluate hpo_test --results data/results_hpo_train --models models/hpo_train --log DEBUG
   ```

##Train, Test and Evaluate with CRFSuite
```
   python src/main.py train --goldstd hpo_train --models models/hpo_train --log DEBUG --entitytype hpo --crf crfsuite
   python src/main.py test --goldstd hpo_test -o pickle data/results_hpo_train --models models/hpo_train --log DEBUG --entitytype hpo --crf crfsuite
   python src/evaluate.py evaluate hpo_test --results data/results_hpo_train --models models/hpo_train --log DEBUG --entitytype hpo
```
##Test and Evaluate for Test Suites
```
   python src/main.py test --goldstd tsuite -o pickle data/results_hpo_train --models models/hpo_train --log DEBUG --entitytype hpo --crf crfsuite
   python src/evaluate.py evaluate tsuite --results data/results_hpo_train --models models/hpo_train --log DEBUG --entitytype hpo 
   ```

Rules can be added to the evaluation parameters:
```
   --rules andor stopwords small_ent twice_validated stopwords gowords posgowords longterms small_len quotes defwords digits lastwords
   ```
# FAQ

##How to run IHP in new, unlabeled, unstructured text?

Replace the sample corpus in corpora/hpo/test_corpus/ by the new, unlabeled, unstructured text and delete the content of corpora/hpo/test_ann/. Then run:
```
    python src/main.py load_corpus --goldstd hpo_test --log DEBUG
    python src/main.py test --goldstd hpo_test -o pickle data/results_hpo_train --models models/hpo_train --log DEBUG
    python src/evaluate.py evaluate hpo_test --results data/results_hpo_train --models models/hpo_train --log DEBUG
```
    
The report file in data/results_hpo_train_report.txt will have the generated annotations marked as false positives (because no annotation file was provided).

## References: 

- M. Lobo, A. Lamurias, and F. Couto, “Identifying human phenotype terms by combining machine learning and validation rules,” BioMed Research International, vol. 2017, pp. 1--14, 2017 (https://doi.org/10.1155/2017/8565739)
