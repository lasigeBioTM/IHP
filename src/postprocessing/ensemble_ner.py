import unicodedata

__author__ = 'Andre'
from sklearn import ensemble
from sklearn.pipeline import Pipeline
import logging
from sklearn.externals import joblib
import os
import cPickle as pickle
import atexit

from text.chemical_entity import chem_words


def word_case(word):
    if word.islower():
        case = 0
    elif word.isupper():
        case = 1
    elif word.istitle():
        case = 2
    else:
        case = 3
    return case


def has_greek_symbol(word):
    for c in word:
        #print c
        try:
            if 'GREEK' in unicodedata.name(c):
                hasgreek = 'HASGREEK'
                return 1
        except ValueError:
            return 0
    return 0


class EnsembleNER(object):
    def __init__(self, path, goldset, base_model, features=None, types=None):
        self.ensemble_pipeline = Pipeline([
            ('clf', ensemble.RandomForestClassifier(criterion="gini", n_estimators=1000))
            ])
        self.base_model = base_model
        self.path = path
        self.predicted = []
        self.res = None
        self.ids, self.data, self.labels = [], [], []
        self.goldset = goldset
        if types: # features is a list of classifier names
            self.types = types
        else:
            self.types = []
        self.feature_names = []
        for t in self.types:
            self.feature_names.append(t)
            self.feature_names.append(t + "_ssm")
        for f in features:
            self.feature_names.append(f)


    def train(self):
        logging.info("training model...")
        self.ensemble_pipeline.fit(self.data, self.labels)

    def test(self):
        logging.info("testing model...")
        self.predicted = self.ensemble_pipeline.predict_proba(self.data)

    def save(self):
        joblib.dump(self.ensemble_pipeline, self.path)
        logging.info("done, saved model as {0.path}".format(self))

    def load(self):
        self.ensemble_pipeline = joblib.load(self.path)

    def generate_data(self, crf_results, corpus="chemdner", supervisioned=True):
        """
        Create the lists necessary to train or test the ensemble classifier
        :param corpus: type of corpus
        :param supervisioned: if true, assign labels values from goldset
        :return:
        """
        self.res = crf_results
        logging.info("generating training data...")
        for did in self.res.corpus.documents:
            for sentence in self.res.corpus.documents[did].sentences:
                for entity in sentence.entities.elist[self.base_model]:
                    vector = []

                    start = entity.dstart
                    end = entity.dend
                    sentence_type = "A"
                    if sentence.sid.endswith("s0"):
                        sentence_type = "T"
                    id = (entity.did, "{0}:{1}:{2}".format(sentence_type, start, end), "1")
                    self.ids.append(id)
                    # 1st set of features: classifiers from the features list and ssm score from each classifier
                    for c in self.types:
                        if c in entity.recognized_by:
                            vector.append(entity.score[c])
                            vector.append(entity.ssm_score_all[c])
                        else:
                            vector.append(0)
                            vector.append(0)
                    #chebi score
                    vector.append(entity.chebi_score)
                    if "case" in self.feature_names:
                        vector.append(word_case(entity.text))
                    if "number" in self.feature_names:
                        if any(c.isdigit() for c in entity.text):
                            vector.append(1)
                        else:
                            vector.append(0)
                    if "greek" in self.feature_names:
                        vector.append(has_greek_symbol(entity.text))
                    if "dashes" in self.feature_names:
                        vector.append(entity.text.count("-"))
                    if "commas" in self.feature_names:
                        vector.append(entity.text.count(","))
                    if "length" in self.feature_names:
                        vector.append(len(entity.text))
                    if "chemwords" in self.feature_names:
                        has_chemwords = 0
                        for w in entity.text.split(" "):
                            if w.lower() in chem_words:
                                has_chemwords = 1
                        vector.append(has_chemwords)

                    # logging.debug("{} - {}".format(entity.text.encode("utf8"), vector))
                    #logging.info(entity.score)
                    self.data.append(vector)
                    if supervisioned:
                        if id in self.goldset:
                            label = 1
                        else:
                            label = 0
                        self.labels.append(label)
                    #if label == 0:
                    #    print id, vector, label

        #print goldset[:10]
        #print features
        #logging.info(self.features)

        logging.info("0: %s; 1: %s", len([x for x in self.labels if x == 0]),
                     len([x for x in self.labels if x == 1]))
        #print ids
        #print [i for i in ids if i[0] == "21826085"]
        #print [g for g in goldset if g[0] == "21826085"]
        #return ids, data, labels
