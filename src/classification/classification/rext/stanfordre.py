# from __future__ import unicode_literals

import codecs
import logging
import os
import platform
import sys
from subprocess import PIPE
from subprocess import Popen
from pycorenlp import StanfordCoreNLP
import itertools

from classification.rext.kernelmodels import ReModel
from config import config


class StanfordRE(ReModel):
    def __init__(self, corpus, relationtype, modelname="stanfordre_classifier.ser"):
        super(StanfordRE, self).__init__()
        self.modelname = modelname
        self.pairs = {}
        self.corenlp_client = None
        self.relationtype = relationtype
        self.corpus = corpus

    def generate_data(self, corpus, modelname, pairtypes):
        if os.path.isfile(self.temp_dir + modelname + ".txt"):
            print "removed old data"
            os.remove(self.temp_dir + modelname + ".txt")
        trainlines = []
        # get all entities of this document
        # doc_entities = []
        pcount = 0
        truepcount = 0
        ns = 0
        for sentence in corpus.get_sentences("goldstandard"):
            logging.info("{}".format(sentence.sid))
            nt_to_entity = {}
            for e in sentence.entities.elist['goldstandard']:
                # TODO: merge tokens of entity
                nt = str(e.tokens[0].order)
                nt_to_entity[nt] = e
            # print nt_to_entity
            # ns = sentence.sid.split("s")[-1]
            for t in sentence.tokens:
                nt = str(t.order)
                # print nt, nt in nt_to_entity
                if nt in nt_to_entity:
                    # print nt, nt_to_entity[nt], nt_to_entity[nt].type
                    #l = [str(ns), nt_to_entity[nt].type, nt, "O", t.pos, t.text, "O", "O", "O"]
                    # TODO: change other to entitiy name
                    l = [str(ns), "Other", nt, "O", t.pos, t.text, "O", "O", "O"]
                else:
                    # print nt, nt_to_entity
                    l = [str(ns), "O", nt, "O", t.pos, t.text, "O", "O", "O"]
                trainlines.append(l)
            trainlines.append([""])
            sentence_entities = [entity for entity in sentence.entities.elist["goldstandard"]]
            # logging.debug("sentence {} has {} entities ({})".format(sentence.sid, len(sentence_entities), len(sentence.entities.elist["goldstandard"])))
            for pair in itertools.combinations(sentence_entities, 2):
                if pair[0].type == pairtypes[0] and pair[1].type == pairtypes[1] or pair[1].type == pairtypes[0] and pair[0].type == pairtypes[1]:
                    # logging.debug(pair)
                    if pair[0].type == pairtypes[0]:
                        e1id = pair[0].eid
                        e2id = pair[1].eid
                    else:
                        e1id = pair[1].eid
                        e2id = pair[0].eid
                        pair = (pair[1], pair[0])
                    pid = sentence.did + ".p" + str(pcount)
                    # self.pairs[pid] = (e1id, e2id)
                    self.pairs[pid] = pair
                    if e2id in pair[0].targets:
                        truepcount += 1
                        nt1 = str(pair[0].tokens[0].order)
                        nt2 = str(pair[1].tokens[0].order)
                        trainlines.append([nt1, nt2, "+".join(pairtypes)])
                pcount += 1
                trainlines.append([""])
                ns += 1



        logging.info("Writing {} lines...".format(len(trainlines)))
        with codecs.open(self.temp_dir + modelname + ".corp", 'w', "utf-8") as trainfile:
            for l in trainlines:
                # print l
                trainfile.write("\t".join(l) + "\n")
        logging.info("True/total relations:{}/{} ({})".format(truepcount, pcount, str(1.0*truepcount/pcount)))

    def write_props(self):
        with open(config.corenlp_dir + "roth.properties", 'r') as propfile:
            lines = propfile.readlines()

        print lines
        with open(config.corenlp_dir + "roth.properties", 'w') as propfile:
            for l in lines:
                if l.startswith("serializedRelationExtractorPath"):
                    propfile.write("serializedRelationExtractorPath = {}\n".format(config.corenlp_dir + self.modelname))
                elif l.startswith("trainPath"):
                    propfile.write("trainPath = {}\n".format(self.temp_dir + self.modelname + ".corp"))
                else:
                    propfile.write(l)

    def train(self):
        self.generate_data(self.corpus, self.modelname, pairtypes=self.relationtype)
        # java -cp classpath edu.stanford.nlp.ie.machinereading.MachineReading --arguments roth.properties
        if os.path.isfile(config.corenlp_dir + self.modelname):
            print "removed old model"
            os.remove(config.corenlp_dir + self.modelname)
        if not os.path.isfile(self.temp_dir + self.modelname  + ".corp"):
            print "could not find training file " + config.corenlp_dir + self.modelname + ".corp"
            sys.exit()
        self.write_props()
        classpath = config.corenlp_dir + "*"
        srecall = ['java', '-mx3g', '-classpath', classpath, "edu.stanford.nlp.ie.machinereading.MachineReading",
                          "--arguments",  config.corenlp_dir + "roth.properties"]
        print " ".join(srecall)
        # sys.exit()
        srecall = Popen(srecall) #, stdout=PIPE, stderr=PIPE)
        res  = srecall.communicate()
        if not os.path.isfile(config.corenlp_dir + self.modelname):
            print "error with StanfordRE! model file was not created"
            print res[1]
            sys.exit()
        else:
            statinfo = os.stat(config.corenlp_dir + self.modelname)
            if statinfo.st_size == 0:
                print "error with StanfordRE! model has 0 bytes"
                print res[0]
                print res[1]
                sys.exit()
        # logging.debug(res)

    def load_classifier(self, inputfile="slk_classifier.model.txt", outputfile="jsre_results.txt"):
        self.corenlp_client = StanfordCoreNLP('http://localhost:9000')
        # sup.relation.model=
        tokenkeys = set()
        sentencekeys = set()
        for d in self.corpus.documents:
            for s in self.corpus.documents[d].sentences:
                corenlpres = self.corenlp_client.annotate(s.text.encode("utf8"), properties={
                        'ssplit.eolonly': True,
                        'openie.triple.all_nominals': True,
                        'openie.triple.strict': False,
                        'openie.max_entailments_per_clause': 500,
                        'annotators': 'tokenize,ssplit,pos,depparse,natlog,openie',
                        #'annotators': 'tokenize, ssplit, pos, lemma, ner, parse, relation, openie',
                        'outputFormat': 'json',
                        # 'sup.relation.model': self.modelname
                    })
                for o in corenlpres["sentences"][0]["openie"]:
                    if "mir" in o["object"] or "mir" in o["subject"]:
                        print "{}={}>{}".format(o["subject"], o["relation"], o["object"])


    def test(self, outputfile="jsre_results.txt"):
        pass

    def get_predictions(self, corpus, examplesfile="slk_classifier.model.txt", resultfile="jsre_results.txt"):
        pass