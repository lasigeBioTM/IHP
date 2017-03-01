#!/usr/bin/env python
#shallow linguistic kernel
import sys, os
import os.path
import xml.etree.ElementTree as ET
import logging
from optparse import OptionParser
import pickle
import operator
from subprocess import Popen, PIPE
from time import time
#from pandas import DataFrame
import numpy as np
from scipy.stats import mode
import platform
import re

import nltk
import nltk.data
from nltk.tree import Tree
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics as skm
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

import relations


basedir = "models/ddi_models/"
temp_dir = "temp/"

def reparse_tree(line):
    ptree = Tree.fromstring(line)
    leaves = ptree.leaves()
    

def get_pair_instances(pair, pairtext):
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


def generatejSRE_line(pairtext, pos, lemmas, ner):
    candidates = [0,0]
    body = ''
    for it in range(len(pairtext)):
        #for it in range(len(pairtext)):
        if pairtext[it] == "#drug-candidatea#":
            #print pairtext[i],
            tokentype = 'DRUG'
            #tokentype = etypes[0]
            tokenlabel = 'A'
            candidates[0] += 1
            tokentext = "#candidate#"
            #tokentext = entitytext[0]
            #tokentext = pairtext[it].lstrip()
            lemma = tokentext
        elif pairtext[it] == "#drug-candidateb#":
            #print pairtext[i]
            tokentype = 'DRUG'
            #tokentype = etypes[0]
            tokenlabel = 'T'
            tokentext = "#candidate#"
            #tokentext = pairtext[it].lstrip()
            #tokentext = entitytext[1]
            lemma = tokentext
            candidates[1] += 1
        elif pairtext[it] == "#drug-entity#":
            tokentype = 'DRUG'
            tokenlabel = 'O'
            tokentext = pairtext[it].lstrip()
            lemma = tokentext
        else:
            # logging.debug("{}".format(pairtext[it].lstrip()))
            tokentype = ner[it]
            tokenlabel = 'O'
            tokentext = pairtext[it].lstrip()
            lemma = lemmas[it]
            if tokentext == '-RRB-':
                tokentext = ')'
                lemma = ')'
            elif tokentext == '-LRB-':
                tokentext = '('
                lemma = '('
        #if ' ' in pairtext[it][0].lstrip() or '\n' in pairtext[it][0].lstrip():
        #    print "token with spaces!"
        #    print pairs[pair][ddi.PAIR_TOKENS][it][0].lstrip()
        #    sys.exit()

        body += "&&".join([str(it), tokentext,
                          lemma,
                          pos[it],
                          tokentype, tokenlabel])
        body += ' '
    #logging.debug("%s\t%s\t%s", str(trueddi), pair, body)
    if candidates[0] == 0:
        logging.debug("missing first candidate on pair ")
        body = "0&&#candidate#&&#candidate#&&-None-&&drug&&T " + body
        #print body
    elif candidates[1] == 0:
        logging.debug("missing second candidate on pair")
        #print body
        body += " " + str(it+1) + "&&#candidate#&&#candidate#&&-None-&&drug&&T "
    return body

def generatejSREdata(pairs, sentence, basemodel, savefile, train=False):
    examplelines = []
    for pair in pairs:
        #logging.debug(pair)
        e1id = pair.eids[0]
        e2id = pair.eids[1]
        sid = sentence.sid

        sentence_tokens = [t.text for t in sentence.tokens]
        #print pairtext,
        if not pair.relation:
            trueddi = 0
        else:
            trueddi = 1

        #print pairtext
        pos = [t.pos for t in sentence.tokens]
        lemmas = [t.lemma for t in sentence.tokens]
        ner = [t.tag for t in sentence.tokens]
        logging.debug("{} {} {} {}".format(len(sentence_tokens), len(pos), len(lemmas), len(ner)))
        pair_text, pos, lemmas, ner = blind_all_entities(sentence_tokens, sentence.entities.elist[basemodel],
                                                         [e1id, e2id], pos, lemmas, ner)
        logging.debug("{} {} {} {}".format(len(pair_text), len(pos), len(lemmas), len(ner)))
        #logging.debug("generating jsre lines...")
        #for i in range(len(pairinstances)):
            #body = generatejSRE_line(pairinstances[i], pos, stems, ner)
        body = generatejSRE_line(pair_text, pos, lemmas, ner)
        examplelines.append(str(trueddi) + '\t' + pair.pid + '.i' + '0\t' + body + '\n')
            #print body
        #elif candidates[0] > 1 or candidates[1] > 1:
        #    print "multiple candidates!!", pairtext
    # logging.debug("writing to file...")
    with open(temp_dir + savefile, 'w') as trainfile:
        for l in examplelines:
            #print l
            trainfile.write(l)
    # logging.info("wrote " + temp_dir + savefile)


def compact_id(eid):
    return eid.replace('.', '').replace('-', '')


def blind_all_entities(tokens, entities, eids, pos, lemmas, ner):
    # logging.info(eids)
    ogtokens = tokens[:]
    found1 = 0
    found2 = 0
    for e in entities:
        if e.eid == eids[0]:
            first_token = e.tokens[0].order + found1 + found2
            # logging.debug("{} {} {} {}".format(tokens[first_token], pos[first_token], lemmas[first_token], ner[first_token]))
            tokens = tokens[:first_token] + ["#drug-candidatea#"] + tokens[first_token:]
            pos = pos[:first_token] + [pos[first_token]] + pos[:first_token]
            lemmas = lemmas[:first_token] + [lemmas[first_token]] + lemmas[:first_token]
            ner = ner[:first_token] + [ner[first_token]] + ner[:first_token]
            # logging.debug("found e1 {} {} {} {}".format(len(tokens), len(pos), len(lemmas), len(ner)))
            found1 += 1
        elif e.eid == eids[1]:
            first_token = e.tokens[0].order + found1 + found2
            # logging.debug("{} {} {} {}".format(tokens[first_token], pos[first_token], lemmas[first_token], ner[first_token]))
            tokens = tokens[:first_token] + ["#drug-candidateb#"] + tokens[first_token:]
            pos = pos[:first_token] + [pos[first_token]] + pos[first_token:]
            lemmas = lemmas[:first_token] + [lemmas[first_token]] + lemmas[first_token:]
            ner = ner[:first_token] + [ner[first_token]] + ner[first_token:]
            # logging.debug("found e2 {} {} {} {}".format(len(tokens), len(pos), len(lemmas), len(ner)))
            found2 += 1
        else:
            tokens[e.tokens[0].order] = "#drug-entity#"
            #print "found other drug"
    if (not found1 or not found2):
        logging.warning("ddi_preprocess: could not find one of the pairs here!")
        logging.warning(tokens)
        logging.warning(ogtokens)
        logging.info([(e.text, e.eid) for e in entities if e.eid in eids])
        sys.exit()
    # logging.debug("{} {} {} {}".format(len(tokens), len(pos), len(lemmas), len(ner)))
    return tokens, pos, lemmas, ner


def trainjSRE(inputfile, model="slk_classifier.model"):
    if os.path.isfile("ddi_models/" + model):
        print "removed old model"
        os.remove("ddi_models/" + model)
    if not os.path.isfile(temp_dir + inputfile):
        print "could not find training file " + basedir + inputfile
        sys.exit()
    if platform.system() == "Windows":
        sep = ";"
    else:
        sep = ":"
    libs = ["libsvm-2.8.jar", "log4j-1.2.8.jar", "commons-digester.jar", "commons-beanutils.jar", "commons-logging.jar", "commons-collections.jar"]
    classpath = 'jsre/jsre-1.1/bin/' + sep + sep.join(["jsre/jsre-1.1/lib/" + l for l in libs])
    jsrecall = ['java', '-mx8g', '-classpath', classpath, "org.itc.irst.tcc.sre.Train",
                      "-k",  "SL", "-n", "4", "-w", "3", "-m", "4098",  "-c", "2",
                      temp_dir + inputfile, basedir + model]
    #print " ".join(jsrecall)
    jsrecall = Popen(jsrecall, stdout = PIPE, stderr = PIPE)
    res  = jsrecall.communicate()
    if not os.path.isfile("ddi_models/" + model):
        print "error with jsre!"
        print res[1]
        sys.exit()
    else:
        statinfo = os.stat("ddi_models/" + model)
        if statinfo.st_size == 0:
            print "error with jsre! model has 0 bytes"
            print res[0]
            print res[1]
            sys.exit()
    #logging.debug(res)


def testjSRE(inputfile, outputfile, model="slk_classifier.model"):
    if os.path.isfile(temp_dir + outputfile):
        os.remove(temp_dir + outputfile)
    if not os.path.isfile(basedir + model):
        print "model", basedir + model, "not found"
        sys.exit()   
    if platform.system() == "Windows":
        sep = ";"
    else:
        sep = ":"
    #logging.debug("testing %s with %s to %s", temp_dir + inputfile,
    #              basedir + model, temp_dir + outputfile)
    libs = ["libsvm-2.8.jar", "log4j-1.2.8.jar", "commons-digester.jar", "commons-beanutils.jar", "commons-logging.jar", "commons-collections.jar"]
    classpath = 'bin/jsre/jsre-1.1/bin/' + sep + sep.join(["bin/jsre/jsre-1.1/lib/" + l for l in libs])
    jsrecommand = ['java', '-mx4g', '-classpath', classpath, "org.itc.irst.tcc.sre.Predict",
                      temp_dir + inputfile, basedir + model, temp_dir + outputfile]
    #print ' '.join(jsrecommand)
    jsrecall = Popen(jsrecommand, stdout = PIPE, stderr = PIPE)
    res = jsrecall.communicate()
    #logging.debug(res[0].strip().split('\n')[-2:])
    #os.system(' '.join(jsrecommand))
    if not os.path.isfile(temp_dir + outputfile):
        print "something went wrong with JSRE!"
        print res
        sys.exit()
    #logging.debug("done.")


def getjSREPredicitons(examplesfile, resultfile, pairs):
    #pred_y = []
    with open(temp_dir + resultfile, 'r') as resfile:
        pred = resfile.readlines()

    with open(temp_dir + examplesfile, 'r') as trainfile:
        original = trainfile.readlines()

    if len(pred) != len(original):
        print "different number of predictions!"
        sys.exit()

    temppreds = {}
    for i in range(len(pred)):
        original_tsv = original[i].split('\t')
        # logging.debug(original_tsv)
        pid = '.'.join(original_tsv[1].split('.')[:-1])
        if pid not in pairs:
            print "pair not in pairs!"
            print pid
            print pairs
            sys.exit()
        #confirm that everything makes sense
        # true = float(original_tsv[0])
        # if true == 0:
        #    true = -1

        p = float(pred[i].strip())
        if p == 0:
            p = -1
        if p == 2:
            print "p=2!"
            p = 1
        logging.debug("{} - {} SLK: {}".format(pairs[pid].entities[0], pairs[pid].entities[1], p))
        #if pair not in temppreds:
        #    temppreds[pair] = []
        #temppreds[pair].append(p)
        pairs[pid].recognized_by[relations.SLK_PRED] = p
    '''for pair in temppreds:
        if relations.SLK_PRED not in pairs[pair]:
            pairs[pair][relations.SLK_PRED] = {}
        p = mode(temppreds[pair])[0][0]
        if len(set(temppreds[pair])) > 1:
            print temppreds[pair], p
        pairs[pair][relations.SLK_PRED][dditype] = p
        #if pairs[pair][ddi.SLK_PRED][dditype] and not pairs[pair][ddi.SLK_PRED]["all"]:
        #    logging.info("type classifier %s found a new true pair: %s", dditype, pair)

    for pair in pairs:
        if relations.SLK_PRED not in pairs[pair]:
            pairs[pair][relations.SLK_PRED] = {}
        if dditype not in pairs[pair][relations.SLK_PRED]:
             pairs[pair][relations.SLK_PRED][dditype] = -1'''
    return pairs

def get_wordnet_pos(treebank_tag):

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


def get_svm_train_line(tree, pair, sid):
    lmtzr = WordNetLemmatizer()
    e1id = compact_id(pair.eids[0])
    e2id = compact_id(pair.eids[1])
    tree = tree.replace(pair.entities[0].tokens[0].text, 'candidatedrug')
    tree = tree.replace(pair.entities[1].tokens[0].text, 'candidatedrug')
    #tree = tree.replace(sid.replace('.', '').replace('-', '') + 'e', 'otherdrug')
    sid2 = compact_id(sid) + 'e'
    # TODO: replace other entities
    #tree = rext.sub(sid2 + r'\d+', 'otherdrug', tree)
    #print "tree2:", tree
    if tree[0] != '(':
        tree = '(S (' + tree + ' NN))'
    #this depends on the version of nlkt
    ptree = Tree.fromstring(tree)
    #ptree = Tree.parse(tree)
    leaves = list(ptree.pos())
    lemmaleaves = []
    for t in leaves:
        pos = get_wordnet_pos(t[1])
        lemma = lmtzr.lemmatize(t[0].lower(), pos)
        lemmaleaves.append(lemma)
    #lemmaleaves = [ for t in leaves)]
    logging.debug("tree:" + tree)
    line = '1 '
    line += '|BT|'  + tree
    #bowline = '(BOW (' + ' *)('.join(lemmaleaves) + ' *)) '
    #ptree = Tree.parse(bowline)
    #ptree = ptree.pprint(indent=-1000)
    #bowline = ptree.replace('\n', ' ')
    #bowline = '|BT| ' + bowline
    #if not bowline.count("otherdrug") > 8:
    #    line += bowline
    #else:
        #print "problem with BOW!"
    #line += bowline
    line += '|ET| '
    
    #i = 1
    #for m in docsp[ddi.PAIR_SSM_VECTOR]:
    #    line += " %s:%s" % (i, m)
    #    i += 1
    #line += " 2:" + str()
    #line += " |EV|"
    line += '\n'
    return line


def trainSVMTK(docs, pairs, dditype, model="svm_tk_classifier.model", excludesentences=[]):
    if os.path.isfile("ddi_models/" + model):
        os.remove("ddi_models/" + model)
    if os.path.isfile("ddi_models/" + model + ".txt"):
        os.remove("ddi_models/" + model + ".txt")

    #docs = use_external_data(docs, excludesentences, dditype)
    xerrors = 0
    with open("ddi_models/" + model + ".txt", 'w') as train:
        #print pairs
        for p in pairs:
            if dditype != "all" and pairs[p][relations.PAIR_DDI] and pairs[p][relations.PAIR_TYPE] != dditype:
                continue
            sid = relations.getSentenceID(p)
            if sid not in excludesentences:
                tree = pairs[p][relations.PAIR_DEP_TREE][:]
                #print "tree1:", tree
                #if len(docs[sid][ddi.SENTENCE_ENTITIES]) > 20:
                    #print line
                #    line = "1 |BT| (ROOT (NP (NN candidatedrug) (, ,) (NN candidatedrug))) |ET|"
                #    xerrors += 1
                #else:
                line = get_svm_train_line(tree, pairs[p], sid,
                                              docs[sid][relations.SENTENCE_PAIRS][p])
                if not pairs[p][relations.PAIR_DDI]:
                    line = '-' + line
                elif pairs[p][relations.PAIR_TYPE] != dditype and dditype != "all":
                    line = '-' + line

                train.write(line)
    #print "tree errors:", xerrors
    svmlightcall = Popen(["./svm-light-TK-1.2/svm-light-TK-1.2.1/svm_learn", "-t", "5",
                          "-L", "0.4", "-T", "2", "-S", "2", "-g", "10",
                          "-D", "0", "-C", "T", basedir + model + ".txt", basedir + model],
                         stdout = PIPE, stderr = PIPE)
    res  = svmlightcall.communicate()
    if not os.path.isfile("ddi_models/" + model):
        print "failed training model " + basedir + model
        print res
        sys.exit()


def testSVMTK(sentence, pairs, pairs_list, model="svm_tk_classifier.model", tag=""):
    if os.path.isfile(basedir + tag + "svm_test_data.txt"):
            os.remove(basedir + tag + "svm_test_data.txt")
    if os.path.isfile(basedir + tag + "svm_test_output.txt"):
            os.remove(basedir + tag + "svm_test_output.txt")
    #docs = use_external_data(docs, excludesentences, dditype)
    #pidlist = pairs.keys()
    total = 0
    with open(temp_dir + tag + "svm_test_data.txt", 'w') as test:
        for pid in pairs:
            sid = pairs[pid].sid
            tree = sentence.parsetree
            
            #if len(docs[sid][ddi.SENTENCE_ENTITIES]) > 30:
                #print line
                #line = reparse_tree(line)
            #    line = "1 |BT| (ROOT (NP (NN candidatedrug) (, ,) (NN candidatedrug))) |ET|\n"
            #    xerrors += 1
            #else:
            line = get_svm_train_line(tree, pairs[pid], sid)
            line = '-' + line
            test.write(line)
            total += 1
    #print "tree errors:", xerrors, "total:", total
    svmtklightargs = ["./bin/svm-light-TK-1.2/svm-light-TK-1.2.1/svm_classify",
                          temp_dir + tag + "svm_test_data.txt",  basedir + model,
                          temp_dir + tag + "svm_test_output.txt"]
    svmlightcall = Popen(svmtklightargs, stdout=PIPE, stderr=PIPE)
    res  = svmlightcall.communicate()
    # logging.debug(res[0].split('\n')[-3:])
    #os.system(' '.join(svmtklightargs))
    if not os.path.isfile(temp_dir + tag + "svm_test_output.txt"):
        print "something went wrong with SVM-light-TK"
        print res
        sys.exit()
    with open(temp_dir + tag + "svm_test_output.txt", 'r') as out:
        lines = out.readlines()
    if len(lines) != len(pairs_list):
        print "check " + tag + "svm_test_output.txt! something is wrong"
        print res
        sys.exit()
    for p, pid in enumerate(pairs):
        score = float(lines[p])
        if float(score) < 0:
            pairs[pid].recognized_by[relations.SST_PRED] = -1
        else:
            pairs[pid].recognized_by[relations.SST_PRED] = 1
        logging.info("{} - {} SST: {}".format(pairs[pid].entities[0], pairs[pid].entities[0], score))
    return pairs


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
    #build_data_frame(docs)
    #if 'parsetree' not in docs['info']:
    #    for doc in docs:
    #        for s in docs[doc]:
    #            docs[doc][s]['parsetree'] = gettree(docs[doc][s]['tokens'])
    #        docs['info'].append('parsetree')

    #trainEvaluatePairs(docs, nfolds=options.nfolds)

    if options.kernel == 'slk':
        generatejSREdata(docs, options.action + '_pairs.txt')
        if options.action == 'train':
            trainjSRE(options.kernel + '_' + options.action + '_pairs.txt')
        elif options.action == 'test':
            testjSRE(options.kernel + '_' +options.action + '_pairs.txt', options.kernel + '_' + "test_results.txt")
    elif options.kernel == 'svmtk':
        generateSVMTKdata(docs)    
        if options.action == 'train':
            trainSVMTK(options.kernel + '_' +options.action + '_pairs.txt')
        elif options.action == 'test':
            testSVMTK(options.kernel + '_' +options.action + '_pairs.txt', options.kernel + '_' + "test_results.txt")

    
        generateSVMTKdata(docs)
    #tokenslist = tokens.strip().replace('\r', '').split('\n')

if __name__ == "__main__":
    main()
