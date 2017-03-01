#!/usr/bin/env python
#shallow linguistic kernel
import sys, os
import os.path
import xml.etree.ElementTree as ET
import logging
from optparse import OptionParser
import pickle
import operator
from time import time
#from pandas import DataFrame
import platform
import re

import nltk
import nltk.data
from nltk.tree import Tree
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

import relations

class ReModel(object):
    def __init__(self):
        self.basedir = "models/kernel_models/"
        self.temp_dir = "temp/"

    def reparse_tree(self, line):
        ptree = Tree.fromstring(line)
        leaves = ptree.leaves()
    

    def get_pair_instances(self, pair, pairtext):
        pairinstances = []
        #if the first candidate has more than one mention, each one is an instance
        if len(pair[relations.PAIR_E1TOKENS]) > 1:
            #logging.debug("%s e1 tokens", len(pairdic[ddi.PAIR_E1TOKENS]))
            #create to instances for this pair
            #print "a", [pairtext[t] for t in pairs[pair][ddi.PAIR_E1TOKENS]]
            #print "b", [pairtext[t] for t in pairs[pair][ddi.PAIR_E2TOKENS]]
            for idx in pair[relations.PAIR_E1TOKENS]:
                temptokens = pairtext[:]
                #for tidx in range(len(pairtext)):
                #    if tidx != idx and pairtext[tidx] == "#drug-candidatea#":
                #        temptokens.append("#drug-entity#")
                #    else:
                #        temptokens.append(pairtext[tidx])
                for index, item in enumerate(temptokens):
                    if index != idx and item == "#drug-candidatea#":
                        temptokens[index] = "#drug-entity#"
                pairinstances.append(temptokens[:])
        else: # otherwise, consider just one instance for now
            pairinstances.append(pairtext[:])

        # if the second candidate has more than one mention, for each one of candidate1 mention,
        # add another instance for each candidate 2 mention
        if len(pairdic[relations.PAIR_E2TOKENS]) > 1:
            #logging.debug("%s e2 tokens", len(pairdic[ddi.PAIR_E2TOKENS]))
            totalinstances = len(pairinstances)
            #logging.debug("duplicating %s sentences", totalinstances)
            for idx in pairdic[relations.PAIR_E2TOKENS]:
                for isent in range(totalinstances):
                    #logging.debug(' '.join(sent))
                    temptokens = pairinstances[isent][:]
                    for index, item in enumerate(temptokens):
                        if index != idx and item == "#drug-candidateb#":
                            temptokens[index] = "#drug-entity#"
                    #for tidx in range(len(sent)):
                    #    if tidx != idx and pairtext[tidx] == "#drug-candidateb#":
                    #        temptokens.append("#drug-entity#")
                    #    else:
                    #        temptokens.append(pairtext[tidx])
                    pairinstances.append(temptokens[:])
            #print pairinstances

        #originallen = len(pairinstances)
        #duplicate number of instances for this pair, switching roles
        #for i in range(originallen):
        #    inverted = pairinstances[i][:]
        #    for index, t in enumerate(inverted):
        #        if t == "#drug-candidatea#":
        #            inverted[i] = "#drug-candidateb#"
        #        elif t == "#drug-candidateb#":
        #            inverted[i] = "#drug-candidatea#"
        #    pairinstances.append(inverted[:])
        return pairinstances

    def compact_id(self, eid):
        return eid.replace('.', '').replace('-', '')


    def blind_all_entities(self, tokens, entities, eids, pos, lemmas, ner):
        # logging.info(eids)
        ogtokens = tokens[:]
        found1 = 0
        found2 = 0
        # logging.debug(tokens)
        for e in entities:
            first_token = e.tokens[0].order # + found1 + found2
            # print first_token, e.text, tokens
            if e.eid == eids[0]:

                # logging.debug("{} {} {} {}".format(tokens[first_token], pos[first_token], lemmas[first_token], ner[first_token]))
                # tokens = tokens[:first_token] + ["#candidatea#"] + tokens[first_token:]
                # tokens = tokens[:first_token] + ["#candidatea#"] + tokens[first_token+1:]
                tokens[first_token] = "#candidatea#"
                #tokens[0] = "#candidatea#"
                # pos = pos[:first_token] + [pos[first_token]] + pos[:first_token]
                # lemmas = lemmas[:first_token] + [lemmas[first_token]] + lemmas[:first_token]
                # ner = ner[:first_token] + [ner[first_token]] + ner[:first_token]
                # logging.debug("found e1 {} {} {} {}".format(len(tokens), len(pos), len(lemmas), len(ner)))
                found1 += 1
            elif e.eid == eids[1]:
                # logging.debug("{} {} {} {}".format(tokens[first_token], pos[first_token], lemmas[first_token], ner[first_token]))
                # tokens = tokens[:first_token] + ["#candidateb#"] + tokens[first_token:]
                # tokens = tokens[:first_token] + ["#candidateb#"] + tokens[first_token+1:]
                tokens[first_token] = "#candidateb#"
                #tokens[-1] = "#candidateb#"
                # pos = pos[:first_token] + [pos[first_token]] + pos[first_token:]
                # lemmas = lemmas[:first_token] + [lemmas[first_token]] + lemmas[first_token:]
                #ner = ner[:first_token] + [ner[first_token]] + ner[first_token:]
                # logging.debug("found e2 {} {} {} {}".format(len(tokens), len(pos), len(lemmas), len(ner)))
                found2 += 1
            #  else:
            #    tokens[first_token] = "#entity#"
                #print "found other drug"
        """if (not found1 or not found2):
            logging.warning("ddi_preprocess: could not find one of the pairs here!" + " ".join(tokens))
            logging.info([(e.text, e.eid) for e in entities if e.eid in eids])"""
        # logging.debug("{} {} {} {}".format(len(tokens), len(pos), len(lemmas), len(ner)))
        return tokens, pos, lemmas, ner

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def get_pair(self, pid, corpus):
        did = '.'.join(pid.split(".")[:-1])
        for p in corpus.documents[did].pairs.pairs:
            if p.pid == pid:
                return p
        print "pid not found: {}".format(pid)

def main():
    parser = OptionParser(usage='train and evaluate ML model for DDI classification based on the DDI corpus')
    parser.add_option("-f", "--file", dest="file",  action="store", default="pairs.pickle",
                      help="Pickle file to load/store the data")
    parser.add_option("-d", "--dir", action="store", dest="dir", type = "string", default="DDICorpus/Test/DDIextraction/MedLine/",
                      help="Corpus directory with XML files")
    parser.add_option("--reload", action="store_true", default=False, dest="reload",
                      help="Reload corpus")
    parser.add_option("--log", action="store", dest="loglevel", type = "string", default="WARNING",
                      help="Log level")
    parser.add_option("--logfile", action="store", dest="logfile", type="string", default="kernel.log",
                      help="Log file")
    parser.add_option("--nfolds", action="store", dest="nfolds", type="int", default=10,
                      help="Number of cross-validation folds")
    parser.add_option("--action", action="store", dest="action", type="string", default="cv",
                      help="cv, train, test, or classify")
    parser.add_option("--kernel", action="store", dest="kernel", type="string", default="slk",
                      help="slk, svmtk")
    (options, args) = parser.parse_args()
    numeric_level = getattr(logging, options.loglevel.upper(), None)


    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    logging.basicConfig(level=numeric_level, format='%(asctime)s %(levelname)s %(message)s')
    #logging.getLogger().setLevel(numeric_level)
    logging.debug("debug test")
    logging.info("info test")
    logging.warning("warning test")


    if options.file in os.listdir(os.getcwd()) and not options.reload:
        print "loading corpus pickle", options.file
        docs = pickle.load(open(options.file, 'rb'))
    else:
        print "loading corpus", options.dir
        docs = relations.loadCorpus(options.dir)
        pickle.dump(docs, open(options.file, 'wb'))

if __name__ == "__main__":
    main()
