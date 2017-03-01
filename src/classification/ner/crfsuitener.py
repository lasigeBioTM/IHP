import logging
import math
import pycrfsuite
import sys

from classification.ner.simpletagger import SimpleTaggerModel, create_entity
from classification.results import ResultsNER


class CrfSuiteModel(SimpleTaggerModel):
    def __init__(self, path, etype, **kwargs):
        super(CrfSuiteModel, self).__init__(path, etype, **kwargs)

    def train(self):
        logging.info("Training model with CRFsuite")
        self.trainer = pycrfsuite.Trainer(verbose=False) #algorithm="l2sgd",
        #logging.info(self.trainer.get(self.trainer))
        #print self.trainer
        #logging.info(self.data)
        #logging.info(self.labels)
        for xseq, yseq in zip(self.data, self.labels):
            self.trainer.append(xseq, yseq)
        self.trainer.set_params({
            'c1': 0.9833333,   # coefficient for L1 penalty
            'c2': 1,  # coefficient for L2 penalty
            #'epsilon': 1e-10,
            # 'c2': 2,
            #'max_iterations': 2500,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': 1,
            'feature.possible_states': 1
        })
        # self.trainer.set_params({
        #     #'c1': 0.98335,   # coefficient for L1 penalty
        #     'period': 500,
        #     'max_iterations': 10000,
        #     'calibration.eta': 0.05,
        #     'calibration.rate': 1,
        #     'calibration.samples': 10000,
        #     'calibration.candidates': 100,
        #     'calibration.max_trials': 200,
        #      # 'c2': 1e-3,  # coefficient for L2 penalty
        #     # 'c2': 2,
        #     #'max_iterations': 2500,  # stop earlier

        #     # include transitions that are possible, but not observed
        #     'feature.possible_transitions': 1,
        #     'feature.possible_states': 1
        # })
        print "training model..."
        self.trainer.train(self.path + ".model")  # output model filename
        logging.info(self.trainer.logparser.last_iteration)
        print "done."


    def load_tagger(self, port=None):
        logging.info("Loading %s" % self.path + ".model")
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(self.path + ".model")

    def test(self, corpus, port=None):
        logging.info("Testing with %s" % self.path + ".model")
        #self.predicted = [tagger.tag(xseq) for xseq in self.data]
        for xseq in self.data:
            #logging.debug(xseq)
            self.predicted.append(self.tagger.tag(xseq))
            self.scores.append([])
            for i, x in enumerate(self.predicted[-1]):
                #logging.debug("{0}-{1}".format(i,x))
                prob = self.tagger.marginal(x, i)
                if math.isnan(prob):
                    print "NaN!!"
                    if x == "other":
                        prob = 0
                    else:
                        print x, xseq[i]
                        #print xseq
                        #print self.predicted[-1]
                        #sys.exit()
                #else:
                #    print prob
                #a = str(prob) + "\t" + str(x) + "\t" + str(i)
                #logging.info(a)

                self.scores[-1].append(prob)
            #logging.info(xseq)
        results = self.process_results(corpus)
        return results

    def process_results(self, corpus):
        results = ResultsNER(self.path)
        results.corpus = corpus
        for isent, sentence in enumerate(self.predicted):
            results = self.process_sentence(sentence, self.sids[isent], results)
        logging.info("found {} entities".format(len(results.entities)))
        return results

    def process_sentence(self, predicted, sid, results):
        sentence = results.corpus.documents['.'.join(sid.split('.')[:-1])].get_sentence(sid)
        if len(predicted) != len(sentence.tokens):
            print "len(predicted) != len(sentence.tokens); {}!={}".format(len(predicted), len(sentence.tokens))
            sys.exit()
        if sentence is None:
            print sid
            print "not found!"
            sys.exit()
        sentence.tagged = predicted
        new_entity = None
        for it, t in enumerate(predicted):
            token = sentence.tokens[it]
            if t == "single":
                single_entity = create_entity(tokens=[token],
                                      sid=sentence.sid, did=sentence.did,
                                      text=token.text, score=1, etype=self.etype)
                eid = sentence.tag_entity(start=token.start, end=token.end, etype=self.etype,
                                            entity=single_entity, source=self.path)
                single_entity.eid = eid
                results.entities[eid] = single_entity # deepcopy
            elif t == "start":
                new_entity = create_entity(tokens=[token],
                                                   sid=sentence.sid, did=sentence.did,
                                                   text=token.text, score=1, etype=self.etype)
            elif t == "middle":
                if not new_entity:
                    logging.info("starting with inside...")
                    new_entity = create_entity(tokens=[token],
                                                   sid=sentence.sid, did=sentence.did,
                                                   text=token.text, score=1, etype=self.etype)
                else:
                    new_entity.tokens.append(token)
            elif t == "end":
                if not new_entity:
                    new_entity = create_entity(tokens=[token],
                                               sid=sentence.sid, did=sentence.did,
                                               text=token.text,
                                               score=1, etype=self.etype)
                    logging.debug("started from a end: {0}".format(new_entity))
                else:
                    new_entity.tokens.append(token)
                    new_entity.text = sentence.text[new_entity.tokens[0].start:new_entity.tokens[-1].end]
                    new_entity.end = new_entity.start + len(new_entity.text)
                    new_entity.dend = new_entity.dstart + len(new_entity.text)

                #logging.info("%s end: %s" % (new_entity.sid, str(new_entity)))
                #logging.debug("found the end: %s", ''.join([t.text for t in new_entity.tokens]))
                eid = sentence.tag_entity(new_entity.tokens[0].start, new_entity.tokens[-1].end, self.etype,
                                          entity=new_entity, source=self.path)
                new_entity.eid = eid
                results.entities[eid] = new_entity # deepcopy
                new_entity = None
                #logging.debug("completed entity:{}".format(results.entities[eid]))
        return results


    def word2features(self, sent, i):
        # adapted from https://github.com/tpeng/python-crfsuite
        #logging.info("SOOOOOOOOOOOO CALLLLLLLLLLLED ************************")
        word = sent[i][0]
        postag = sent[i][1]
        features = [
            'bias',
            'word.lower=' + word.lower(),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit(),
            'postag=' + postag,
            'postag[:2]=' + postag[:2],
        ]
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.extend([
                '-1:word.lower=' + word1.lower(),
                '-1:word.istitle=%s' % word1.istitle(),
                '-1:word.isupper=%s' % word1.isupper(),
                '-1:postag=' + postag1,
                '-1:postag[:2]=' + postag1[:2],
            ])
        else:
            features.append('BOS')

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.extend([
                '+1:word.lower=' + word1.lower(),
                '+1:word.istitle=%s' % word1.istitle(),
                '+1:word.isupper=%s' % word1.isupper(),
                '+1:postag=' + postag1,
                '+1:postag[:2]=' + postag1[:2],
            ])
        else:
            features.append('EOS')

        return features


    def sent2features(self, sent):
        logging.info("SOOOOOOOOOOOO CALLLLLLLLLLLED ************************")
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        logging.info("SOOOOOOOOOOOO CALLLLLLLLLLLED ************************")
        return [label for token, postag, label in sent]

    def sent2tokens(self, sent):
        logging.info("SOOOOOOOOOOOO CALLLLLLLLLLLED ************************")
        return [token for token, postag, label in sent]
