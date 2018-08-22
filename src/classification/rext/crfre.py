import codecs
import logging
import math
import word2vec
import itertools
import pycrfsuite
import sys

from classification.ner.simpletagger import SimpleTaggerModel, create_entity, feature_extractors
from classification.results import ResultsNER, ResultsRE
from classification.rext.kernelmodels import ReModel
from config import config
from text.pair import Pairs
from word2vec_experiments import load_tair_relations


class CrfSuiteRE(ReModel):
    def __init__(self, corpus, ptype, test=False, modelname="crfre_classifier"):
        super(CrfSuiteRE, self).__init__()
        self.data = []
        self.labels = []
        self.scores = []
        self.predicted = []
        self.entities = []
        self.pairtype = ptype
        self.modelname = ptype + "_" + modelname
        self.gold_relations = set()
        self.tair_pairs = load_tair_relations()
        self.vecmodel = word2vec.load("corpora/Thaliana/documents-processed" + '.bin')
        with codecs.open("seedev_relation.txt", 'r', 'utf-8') as relfile:
            for r in relfile:
                self.gold_relations.add(r.strip())
        self.clusters = word2vec.load_clusters("corpora/Thaliana/documents-processed-clusters.txt")
        #with codecs.open("corpora/Thaliana/documents-clusters.txt", "r", "utf-8") as clusterfile:
        #    for l in clusterfile:
        #        values = l.strip().split(" ")
        #        self.clusters[values[0]] = values[1]
        self.generate_data(corpus, self.modelname, ptype, test)

    def train(self):
        logging.info("Training model with CRFsuite")
        self.trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(self.data, self.labels):
            self.trainer.append(xseq, yseq)
        self.trainer.set_params({
            'c1': 1,   # coefficient for L1 penalty
             # 'c2': 1e-3,  # coefficient for L2 penalty
            # 'c2': 2,
            'max_iterations': 1000,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': False
        })
        print("training model...")
        self.trainer.train(self.modelname + ".model")  # output model filename
        print("done.")


    def load_classifier(self, port=None):
        logging.info("Loading %s" % self.modelname + ".model")
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(self.modelname + ".model")

    def test(self, port=None):
        logging.info("Testing with %s" % self.modelname + ".model")
        #self.predicted = [tagger.tag(xseq) for xseq in self.data]
        for xseq in self.data:
            #logging.debug(xseq)
            self.predicted.append(self.tagger.tag(xseq))
            self.scores.append([])
            for i, x in enumerate(self.predicted[-1]):
                #logging.debug("{0}-{1}".format(i,x))
                prob = self.tagger.marginal(x, i)
                if math.isnan(prob):
                    print("NaN!!")
                    if x == "other":
                        prob = 0
                    else:
                        print(x, xseq[i])
                        #print xseq
                        #print self.predicted[-1]
                        #sys.exit()
                #else:
                #    print prob
                self.scores[-1].append(prob)

    def generate_data(self, corpus, modelname, pairtypes, test):
       # TODO: remove old model
        pcount = 0
        truepcount = 0
        ns = 0
        for did in corpus.documents:
            #doc_entities = corpus.documents[did].get_entities("goldstandard")
            for sentence in corpus.documents[did].sentences:
                if "goldstandard" not in sentence.entities.elist:
                    continue
            # logging.info("{}".format(sentence.sid))
                sentence_entities = [e for e in sentence.entities.elist["goldstandard"] if e.type in config.pair_types[self.pairtype]["source_types"]]
                # logging.debug("sentence {} has {} entities ({})".format(sentence.sid, len(sentence_entities), len(sentence.entities.elist["goldstandard"])))
                #for pair in itertools.permutations(doc_entities, 2):
                #for e in doc_entities:
                    #sn1 = int(pair[0].sid.split(".")[-1][1:])
                    #sn2 = int(pair[1].sid.split(".")[-1][1:])
                    #if abs(sn2 - sn1) > 0 or pair[0].start == pair[1].start or pair[0].end == pair[1].end:
                    #    continue
                    #if self.pairtype in ("Has_Sequence_Identical_To", "Is_Functionally_Equivalent_To") and pair[0].type != pair[1].type:
                    #    continue
                    # logging.info("{}=>{}|{}=>{}".format(pair[0].type, pair[1].type, pairtypes[0], pairtypes[1]))
                if not any([e.type in config.pair_types[self.pairtype]["source_types"] for e in sentence.entities.elist["goldstandard"]]):
                    logging.info("skipped sentence without targets")
                    continue
                for e in sentence_entities:
                    # TODO: ignore if none of the other entities are targets
                    pid = did + ".p" + str(pcount)
                    #sentence = corpus.documents[did].get_sentence(e.sid)
                    # consider only one sentence
                    sentence_features = self.generate_sentence_features(sentence, e)
                    token_entities, entities_order = self.get_token_entities(sentence)
                    if not test:
                        sentence_labels = self.generate_sentence_labels(sentence, e, token_entities)
                    else:
                        sentence_labels = None
                    if test or "B-TARGET" in sentence_labels or "I-TARGET" in sentence_labels or "B-NOTINTERACTING" in sentence_labels or "I-NOTINTERACTING" in sentence_labels:
                        self.labels.append(sentence_labels)
                        self.data.append(sentence_features)
                        self.entities.append((e, entities_order[:]))

    def get_token_entities(self, sentence):
        # TODO: restrict to targets
        token_entities = {}
        entities_order = []
        for e in sentence.entities.elist["goldstandard"]:
            for t in e.tokens:
                if t.order not in token_entities:
                    token_entities[t.order] = []
                token_entities[t.order].append(e)
        for it, t in enumerate(sentence.tokens):
            entities_order.append(token_entities.get(it))
        return token_entities, entities_order

    def generate_sentence_labels(self, sentence, source, token_entities):
        labels = []
        lastlabel = None
        # print (pair[1].eid, self.pairtype), pair[0].targets
        for it, t in enumerate(sentence.tokens):

            if it in token_entities:
                # start label
                if (token_entities[it][0].eid, self.pairtype) in source.targets:
                    if lastlabel is None or lastlabel == "O" or lastlabel.endswith("NOTINTERACTING"):
                        thislabel = "B-TARGET"

                    else:
                        thislabel = "I-TARGET"
                else:
                    thislabel = "O"
                """elif token_entities[it][0].type in config.pair_types[self.pairtype]["target_types"]:
                    if lastlabel is None or lastlabel == "O" or lastlabel.endswith("TARGET"):
                        thislabel = "B-NOTINTERACTING"
                    else:
                        thislabel = "I-NOTINTERACTING"
                else:
                    thislabel = "O"""""
                """for et in token_entities[it][1:]:
                    if (et.eid, self.pairtype) in source.targets:
                        if lastlabel is None or lastlabel == "O" or lastlabel.endswith("NOTINTERACTING"):
                            thislabel += "+B-TARGET"
                        else:
                            thislabel += "I-TARGET"
                    elif et.type in config.pair_types[self.pairtype]["target_types"]:
                        if lastlabel is None or lastlabel == "O" or lastlabel.endswith("TARGET"):
                            thislabel += "+B-NOTINTERACTING"
                        else:
                            thislabel += "+I-NOTINTERACTING"
                    else:
                        thislabel += "+O" """
            else:
                thislabel = "O"
            # if thislabel != "O":
            #     print thislabel
            labels.append(thislabel)
            lastlabel = thislabel
        return labels


    def generate_sentence_features(self, sentence, source):
        features = []
        for it, t in enumerate(sentence.tokens):
            if t in source.tokens:
                entity_type = source.type
                role = "ROLE=SOURCE"
            elif "goldstandard" in t.tags:
                entity_type = t.tags["goldstandard_subtype"]
                if entity_type in config.pair_types[self.pairtype]["target_types"]:
                    """if t.tags["goldstandard"] == "single":
                        matchtext = t.text
                    else:
                        matchtext = t.text
                        tcount = 1
                        nextt = sentence.tokens[it+tcount]
                        while "goldstandard_{}".format(entity_type) in nextt.tags:
                            matchtext += nextt.text
                            tcount += 1
                            nextt = sentence.tokens[it+tcount]
                    rel_text = "{}#{}\t{}\t{}#{}".format(source.type, source.text, self.pairtype, entity_type, matchtext)
                    if rel_text in self.gold_relations:
                        print "found match!", rel_text
                        role = "ROLE=GOLDTARGET"
                    else:"""
                    role = "ROLE=TARGET"
                else:
                    role = "ROLE=ENTITY"
            else:
                entity_type = t.tag
                role = "ROLE=OTHER"
            clusterf = ""
            """if t.lemma.isalnum() and not t.lemma.isnum() and t.lemma.lower() in self.vecmodel.vocab:
                #clusterf = "cluster-" + str(self.clusters[t.text.lower()])
                indexes, metrics = self.vecmodel.cosine(t.text.lower(), n=5)
                simwords = self.vecmodel.generate_response(indexes, metrics).tolist()
                for w in simwords:
                    clusterf += "cluster-" + w[0] + " "
                # print t.text, clusterf
            else:
                clusterf = "cluster-0"""""
                # print t.text, clusterf

            """if t.text.lower() in self.clusters.vocab:
                clusterf = "cluster-" + str(self.clusters[t.text.lower()])
                print clusterf"""
            """else:
                for k in self.clusters.vocab:
                    print k, t.text
                    if k.startswith(t.text):
                        clusterf = "cluster-" + str(self.clusters[k])"""
            token_features = [entity_type, role] #, clusterf]
            existis_in_db = False
            """for tk in self.tair_pairs:
                if tk == source: #or tk.startswith(source.tokens[0].text):
                    for target in self.tair_pairs[tk]:
                        if target[1] == self.pairtype: # and target[0].startswith(t.text):
                            existis_in_db = True
                            print tk, target"""
            #token_features = []
            for f in feature_extractors:
                 token_features.append(feature_extractors[f](sentence, it))
            features.append(token_features)
        # if "B-TARGET" in labels:
        #    print [f[:3] for f in features if "ROLE=OTHER" not in f]
        #    print labels
        return features


    def get_predictions(self, corpus):
        results = ResultsRE(self.modelname)
        temppreds = {}
        for i in range(len(self.entities)):
            # did = ".".join(self.pairs[i][0].sid.split(".")[:-1])
            # pid = did + ".p" + str(i)
            # if "B-TARGET" in self.predicted[i]:
            #     print self.predicted[i]
            # print self.scores
            did = self.entities[i][0].did
            if did not in results.document_pairs:
                    results.document_pairs[did] = Pairs()
            for it, label in enumerate(self.predicted[i]):
                if label.endswith("B-TARGET"):
                    # print self.entities[i][0].text, [(e.text, e.type) for e in self.entities[i][1][it]]
                    for target in self.entities[i][1][it]:
                        pid = did + ".p" + str(i)
                        # if self.pred[i]:
                        #     did = '.'.join(pid.split(".")[:-1])
                        if did not in results.document_pairs:
                            results.document_pairs[did] = Pairs()
                        pair = corpus.documents[did].add_relation(self.entities[i][0], target, self.pairtype, relation=True)
                        results.document_pairs[did].add_pair(pair, "crf")
                        #pair = self.get_pair(pid, corpus)
                        #results.pairs[pid] = pair

                        # logging.debug("{} - {} SLK: {}".format(pair.entities[0], pair.entities[1], p))
                        #if pair not in temppreds:
                        #    temppreds[pair] = []
                        #temppreds[pair].append(p)
                        results.pairs[pid] = pair
                        results.pairs[pid].recognized_by["crf"] = 1
        results.corpus = corpus
        return results


