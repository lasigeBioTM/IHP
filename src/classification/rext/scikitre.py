import os
import logging

import itertools
import word2vec
import numpy as np
import sys
from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

from classification.results import ResultsRE
from classification.rext.kernelmodels import ReModel
from config import config
from text.pair import Pairs
from text.sentence import Sentence


class ScikitRE(ReModel):
    def __init__(self, corpus, relationtype, modelname="scikit_classifier"):
        super(ScikitRE, self).__init__()
        self.modelname = relationtype + "_" + modelname
        self.relationtype = relationtype
        self.pairtype = relationtype
        self.corpus = corpus
        self.pairs = []
        self.features = []
        self.labels = []
        self.pred = []
        self.clusters = word2vec.load_clusters("corpora/Thaliana/documents-processed-clusters.txt")
        self.posfmeasure = make_scorer(f1_score, average='binary', pos_label=True)
        self.generate_data(corpus, modelname, relationtype)
        self.text_clf = Pipeline([('vect', CountVectorizer(analyzer='char_wb', ngram_range=(3,20), min_df=0.0, max_df=0.7)),
                                  #('vect', CountVectorizer(ngram_range=(1,3), binary=False, max_features=None)),
                                  #('tfidf', TfidfTransformer(use_idf=True, norm="l2")),
                                  #('clf', SGDClassifier(loss='hinge', penalty='l1', alpha=0.0001, n_iter=5, random_state=42)),
                                  #('clf', SGDClassifier())
                                  #('clf', svm.NuSVC(nu=0.01 ))
                                   #('clf', RandomForestClassifier(class_weight={False:1, True:2}, n_jobs=-1))
                                  ('clf', MultinomialNB(alpha=0.01, fit_prior=False))
                                  #('clf', DummyClassifier(strategy="constant", constant=True))
                                 ])

    def generate_data(self, corpus, modelname, pairtypes):
       # TODO: remove old model
        pcount = 0
        truepcount = 0
        ns = 0
        for did in corpus.documents:
            doc_entities = corpus.documents[did].get_entities("goldstandard")
            examplelines = []
            # logging.info("{}".format(sentence.sid))
            # sentence_entities = sentence.entities.elist["goldstandard"]
            # logging.debug("sentence {} has {} entities ({})".format(sentence.sid, len(sentence_entities), len(sentence.entities.elist["goldstandard"])))
            for pair in itertools.permutations(doc_entities, 2):
                sn1 = int(pair[0].sid.split(".")[-1][1:])
                sn2 = int(pair[1].sid.split(".")[-1][1:])
                # if self.pairtype in corpus.type_sentences and pair[0].sid not in corpus.type_sentences[self.pairtype]:
                #     continue
                if abs(sn2 - sn1) > 0 or pair[0].start == pair[1].start or pair[0].end == pair[1].end:
                    continue
                # if self.pairtype in ("Has_Sequence_Identical_To", "Is_Functionally_Equivalent_To") and pair[0].type != pair[1].type:
                #     continue
                #if pair[0].text == pair[1].text:
                #    continue
                # logging.info("{}=>{}|{}=>{}".format(pair[0].type, pair[1].type, pairtypes[0], pairtypes[1]))
                if pair[0].type in config.pair_types[self.pairtype]["source_types"] and pair[1].type in config.pair_types[self.pairtype]["target_types"]:
                #if pair[0].type in config.event_types[self.pairtype]["source_types"] and pair[1].type in config.event_types[self.pairtype]["target_types"]:
                                        #pair[1].type in config.pair_types[self.pairtype]["source_types"] and pair[0].type in config.pair_types[self.pairtype]["target_types"]:
                    # logging.debug(pair)
                    #if pair[0].type not in config.pair_types[self.pairtype]["source_types"]:
                    #    pair = (pair[1], pair[0])
                    pid = did + ".p" + str(pcount)
                    # self.pairs[pid] = (e1id, e2id)
                    if sn1 != sn2:
                        sentence1 = corpus.documents[did].get_sentence(pair[0].sid)
                        sentence2 = corpus.documents[did].get_sentence(pair[1].sid)
                        sentence = Sentence(text = sentence1.text + " " + sentence2.text, offset=sentence1.offset)
                        sentence.tokens = sentence1.tokens + sentence2.tokens
                        for t in pair[1].tokens:
                            t.order += len(sentence1.tokens)
                    else:
                        sentence = corpus.documents[did].get_sentence(pair[0].sid)
                    f, label = self.generate_features(sentence, pair)
                    self.features.append(f)
                    self.labels.append(label)
                    self.pairs.append(pair)

    def generate_features(self, sentence, pair):
        if (pair[1].eid, self.pairtype) in pair[0].targets:
        #if any((pair[1].eid, pt) in pair[0].targets for pt in config.event_types[self.pairtype]["subtypes"]):
            label = True
        else:
            label = False
        #f = sentence.text[pair[0].end:pair[1].start] + " " + ' '.join([t.pos for t in sentence.tokens])
        start, end = pair[0].tokens[-1].order, pair[1].tokens[0].order
        token_order1 = [t.order for t in pair[0].tokens]
        token_order2 = [t.order for t in pair[1].tokens]
        order = "###normalorder###"
        entitytext = [pair[0].text, pair[1].text]
        if start > end:
            order = "###reverseorder###"
            start, end = pair[1].tokens[-1].order, pair[0].tokens[0].order
            entitytext = [pair[1].text, pair[0].text]
        #text = [t.lemma + "-" + t.pos for t in sentence.tokens[start:end]]
        # exclude entities from text
        #text = [t.lemma + "==" + t.pos for t in sentence.tokens[start:end] if "goldstandard" not in t.tags]# + ["type1" + pair[0].type,
                                                                             # "type2" + pair[1].type, order]
        #text = [t.lower() for t in text]
        tokens_text = [t.text for t in sentence.tokens]
        #tokens_text = [t.text for t in sentence.tokens[start-5:start+5]]
        #tokens_text += [t.text for t in sentence.tokens[end-5:end+5]]
        stokens = []
        for it, t in enumerate(sentence.tokens):
            #print tokens_text[:it], tokens_text[it:]
            if "-LRB-" in tokens_text[:it] and "-RRB-" in tokens_text[it:] and "-RRB-" not in tokens_text[:it] and "-LRB-" not in tokens_text[it:]:
                #if "(" in t.text or ")" in t.text:
                # print "skipped between ()", t.text
                continue
            elif t.lemma.isdigit():
                # print "digit:", t.lemma, t.text
                continue
            elif t.text == "-LRB-" or t.text == "-RRB-":
                continue
            elif "goldstandard" in t.tags and (len(stokens) == 0 or stokens[-1] != t.tags["goldstandard_subtype"]):
                 stokens.append(t.tags["goldstandard_subtype"])
            #elif not t.text.isalpha():
            #    print "not alpha:", t.text
            #    continue
            else:
                cluster = 0
                """try:
                    cluster = self.clusters[t.text]
                    #print cluster
                except KeyError:
                    cluster = 0"""
                #stokens.append(t.pos + "-" + t.lemma)
                stokens.append(t.pos + "-" + t.lemma)
        #text.append("entity1-" + pair[0].type)
        #text.append("entity2-" + pair[1].type)
        #text.append("=".join([t.pos for t in sentence.tokens[start:end]]))
        f = ' '.join(stokens)
        # print f, label
        return f, label

    def train(self):
        parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1,3), (2,3)],
                      #'vect__binary': (True, False),

                      'clf__alpha': (1e-2, 1e-3, 1e-1, 1e-4, 1e-5),
                      'clf__loss': ('hinge', 'log'),
                      'clf__penalty': ('l2', 'l1', 'elasticnet')

                       # 'clf__nu': (0.5,0.6),
                      #'clf__kernel': ('rbf', 'linear', 'poly'),
                      # 'clf__tol': (1e-3, 1e-4, 1e-2, 1e-4)

                      #'clf__n_estimators': (10, 50, 100, 500),
                      #'clf__criterion': ('gini', 'entropy'),
                      #'clf__max_features': ("auto", "log2", 100,)

                     #'clf__alpha': (0, 1e-2, 1e-3, 1e-1, 1e-4, 1e-5),
                      #'clf__fit_prior': (False, True),
                     }
        # gs_clf = GridSearchCV(self.text_clf, parameters, n_jobs=-1, scoring=self.posfmeasure)
        # gs_clf = gs_clf.fit(self.features, self.labels)
        # print gs_clf.best_params_
        logging.info("Traning with {}/{} true pairs".format(str(sum(self.labels)), str(len(self.labels))))
        try:
            self.text_clf = self.text_clf.fit(self.features, self.labels)
        except ValueError:
            print "error training {}".format(self.modelname)
            return
        if not os.path.exists(self.basedir + self.modelname):
            os.makedirs(self.basedir + self.modelname)
        logging.info("Training complete, saving to {}/{}/{}.pkl".format(self.basedir, self.modelname, self.modelname))
        joblib.dump(self.text_clf, "{}/{}/{}.pkl".format(self.basedir, self.modelname, self.modelname))
        ch2 = SelectKBest(chi2, k=20)
        half_point = int(len(self.features)*0.5)
        X_train = self.text_clf.named_steps["vect"].fit_transform(self.features[:half_point])
        X_test = self.text_clf.named_steps["vect"].transform(self.features[half_point:])
        X_train = ch2.fit_transform(X_train, self.labels[:half_point])
        X_test = ch2.transform(X_test)
        feature_names = self.text_clf.named_steps["vect"].get_feature_names()
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
        print feature_names
        # joblib.dump(gs_clf.best_estimator_, "{}/{}/{}.pkl".format(self.basedir, self.modelname, self.modelname))
        # self.test()

    def load_classifier(self):
        self.text_clf = joblib.load("{}/{}/{}.pkl".format(self.basedir, self.modelname, self.modelname))

    def test(self):
        self.pred = self.text_clf.predict(self.features)

        # for doc, category in zip(self.features, self.pred):
        #     print '%r => %s' % (doc, category)
        print np.mean(self.pred == self.labels)
        print(metrics.classification_report(self.labels, self.pred))

    def get_predictions(self, corpus):
        #real_pair_type = config.event_types[self.pairtype]["subtypes"][0]
        results = ResultsRE(self.modelname)
        temppreds = {}
        for i in range(len(self.pred)):
            did = ".".join(self.pairs[i][0].sid.split(".")[:-1])
            pid = did + ".p" + str(i)
            if self.pred[i]:
                did = '.'.join(pid.split(".")[:-1])
                if did not in results.document_pairs:
                    results.document_pairs[did] = Pairs()
                #pair = corpus.documents[did].add_relation(self.pairs[i][0], self.pairs[i][1], real_pair_type, relation=True)
                pair = corpus.documents[did].add_relation(self.pairs[i][0], self.pairs[i][1], self.pairtype, relation=True)
                results.document_pairs[did].add_pair(pair, "scikit")
                #pair = self.get_pair(pid, corpus)
                results.pairs[pid] = pair

                # logging.debug("{} - {} SLK: {}".format(pair.entities[0], pair.entities[1], p))
                #if pair not in temppreds:
                #    temppreds[pair] = []
                #temppreds[pair].append(p)
                results.pairs[pid].recognized_by["scikit"] = 1
        results.corpus = corpus
        return results

