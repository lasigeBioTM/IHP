from __future__ import unicode_literals
import os
import logging
import random
import sys
from classification.rext.kernelmodels import ReModel
from subprocess import Popen, PIPE
import platform
import itertools
import codecs
from classification.results import ResultsRE
from config import config
from text.pair import Pairs


class JSREKernel(ReModel):

    def __init__(self, corpus, relationtype, modelname="ratio_slk_classifier.model", train=False):
        super(JSREKernel, self).__init__()
        self.modelname = relationtype + "_" + modelname
        self.pairtype = relationtype
        self.test_jsre = []
        self.pairs = {}
        self.resultsfile = None
        self.examplesfile = None
        self.generatejSREdata(corpus, train=train, pairtype=relationtype)

    def load_classifier(self, outputfile="jsre_results.txt"):
        self.resultsfile = self.temp_dir + self.pairtype + "_" + outputfile
        self.examplesfile = self.temp_dir + self.modelname + ".txt"
        if os.path.isfile(self.temp_dir + self.pairtype + "_" + outputfile):
            os.remove(self.temp_dir + self.pairtype + "_" + outputfile)
        if not os.path.isfile(self.basedir + self.modelname):
            print "model", self.basedir + self.modelname, "not found"
            sys.exit()
        if platform.system() == "Windows":
            sep = ";"
        else:
            sep = ":"
        #logging.debug("testing %s with %s to %s", temp_dir + inputfile,
        #              basedir + model, temp_dir + outputfile)
        libs = ["libsvm-2.8.jar", "log4j-1.2.8.jar", "commons-digester.jar", "commons-beanutils.jar", "commons-logging.jar", "commons-collections.jar"]
        classpath = 'bin/jsre/jsre-1.1/bin'+ sep + sep.join(["bin/jsre/jsre-1.1/lib/" + l for l in libs])
        self.test_jsre = ['java', '-mx4g', '-classpath', classpath, "org.itc.irst.tcc.sre.Predict",
                          self.examplesfile, self.basedir + self.modelname,
                          self.resultsfile]
        #print ' '.join(jsrecommand)

    def train(self):
        if os.path.isfile(self.basedir + self.modelname):
            print "removed old model"
            os.remove(self.basedir + self.modelname)
        if not os.path.isfile(self.temp_dir + self.modelname  + ".txt"):
            print "could not find training file " + self.basedir + self.modelname + ".txt"
            sys.exit()
        if platform.system() == "Windows":
            sep = ";"
        else:
            sep = ":"
        libs = ["libsvm-2.8.jar", "log4j-1.2.8.jar", "commons-digester.jar", "commons-beanutils.jar",
                "commons-logging.jar", "commons-collections.jar"]
        classpath = 'bin/jsre/jsre-1.1/bin/' + sep + sep.join(["bin/jsre/jsre-1.1/lib/" + l for l in libs])
        jsrecall = ['java', '-mx8g', '-classpath', classpath, "org.itc.irst.tcc.sre.Train",
                          "-k",  "SL", "-n", "2", "-w", "2", "-m", "3072",  # "-c", str(2),
                          self.temp_dir + self.modelname + ".txt", self.basedir + self.modelname]
        # print " ".join(jsrecall)
        jsrecall = Popen(jsrecall, stdout=PIPE, stderr=PIPE)
        res  = jsrecall.communicate()
        if not os.path.isfile(self.basedir + self.modelname):
            print "error with jsre! model file was no created"
            print res[1]
            sys.exit()
        else:
            statinfo = os.stat(self.basedir + self.modelname)
            if statinfo.st_size == 0:
                print "error with jsre! model has 0 bytes"
                print res[0]
                print res[1]
                sys.exit()
        #logging.debug(res)


    def test(self, outputfile="jsre_results.txt"):
        # print " ".join(self.test_jsre)
        jsrecall = Popen(self.test_jsre, stdout=PIPE, stderr=PIPE)
        res = jsrecall.communicate()
        #logging.debug(res[0].strip().split('\n')[-2:])
        #os.system(' '.join(jsrecommand))
        if not os.path.isfile(self.temp_dir + outputfile):
            print "something went wrong with JSRE!"
            print res
            sys.exit()
        logging.debug("done.")

    def get_sentence_instance(self, sentence, e1id, e2id, pair):

        tokens = [t for t in sentence.tokens]
        #start, end = pair[0].tokens[0].order, pair[1].tokens[-1].order
        #if pair[0].tokens[0].order > pair[1].tokens[-1].order:
        #    start, end = end, start
        #tokens = [t for t in sentence.tokens[start:end]]
        tokens_text = [t.text for t in tokens]
        # print tokens_text
        pos = [t.pos for t in tokens]
        lemmas = [t.lemma for t in tokens]
        ner = [t.tag for t in tokens]
        #logging.debug("{} {} {} {}".format(len(tokens1), len(pos), len(lemmas), len(ner)))
        return self.blind_all_entities(tokens_text, sentence.entities.elist["goldstandard"],
                                       [e1id, e2id], pos, lemmas, ner)

    def generatejSREdata(self, corpus, train=False, pairtype="all"):
        if os.path.isfile(self.temp_dir + self.modelname + ".txt"):
            print "removed old data"
            os.remove(self.temp_dir + self.modelname + ".txt")
        examplelines = []
        # get all entities of this document
        # doc_entities = []
        pairtypes = (config.pair_types[pairtype]["source_types"], config.pair_types[pairtype]["target_types"])
        # pairtypes = (config.event_types[pairtype]["source_types"], config.event_types[pairtype]["target_types"])
        pcount = 0
        truepcount = 0
        strue = 0
        sfalse = 0
        skipped = 0
        for sentence in corpus.get_sentences("goldstandard"):
        #for did in corpus.documents:
            did = sentence.did
            #doc_entities = corpus.documents[did].get_entities("goldstandard")
            examplelines = []
            pos_sentences = set()
            sids = []
            # print len(corpus.type_sentences[pairtype])
            sentence_entities = [entity for entity in sentence.entities.elist["goldstandard"]]
            # logging.debug("sentence {} has {} entities ({})".format(sentence.sid, len(sentence_entities), len(sentence.entities.elist["goldstandard"])))
            for pair in itertools.permutations(sentence_entities, 2):
                # print pair[0].type, pair[1].type, pairtypes
                sid1 = pair[0].eid.split(".")[-2]
                sid2 = pair[1].eid.split(".")[-2]
                # if pairtype in corpus.type_sentences and pair[0].sid not in corpus.type_sentences[pairtype]:
                #     continue
                sids.append((pair[0].sid, pair[0].sid))
                sn1 = int(sid1[1:])
                sn2 = int(sid2[1:])
                if pair[0].start == pair[1].start or pair[0].end == pair[1].end:
                    continue
                if pairtype in ("Has_Sequence_Identical_To", "Is_Functionally_Equivalent_To") and pair[0].type != pair[1].type:
                    continue
                if pair[0].type in pairtypes[0] and pair[1].type in pairtypes[1]: # or\
                   # pair[1].type in pairtypes[0] and pair[0].type in pairtypes[1]:
                    # logging.debug(pair)
                    """if pair[0].type == pairtypes[0]:

                    else:
                        e1id = pair[1].eid
                        e2id = pair[0].eid
                        pair = (pair[1], pair[0])"""
                    # print e1id, e2id
                    e1id = pair[0].eid
                    e2id = pair[1].eid
                    pid = did + ".p" + str(pcount)
                    # self.pairs[pid] = (e1id, e2id)
                    self.pairs[pid] = pair
                    # sentence1 = corpus.documents[did].get_sentence(pair[0].sid)
                    #sentence1 = sentence
                    # logging.info("{}  {}-{} => {}-{}".format(sentence.sid, e1id, pair[0].text, e2id, pair[1].text))
                    #sentence = corpus.documents[did].get_sentence(did + "." + sid1)
                    tokens_text, pos, lemmas, ner = self.get_sentence_instance(sentence, e1id, e2id, pair)
                    # print tokens_text, pair[0].text, pair[0].start, pair[1].text, pair[1].start, pair[0].start == pair[1].start
                    # logging.debug("{} {} {} {}".format(len(pair_text), len(pos), len(lemmas), len(ner)))
                    #logging.debug("generating jsre lines...")
                    #for i in range(len(pairinstances)):
                        #body = generatejSRE_line(pairinstances[i], pos, stems, ner)

                    trueddi = 0
                    #print (e2id, pairtype), pair[0].targets
                    if (e2id, pairtype) in pair[0].targets:
                    #if any((pair[1].eid, pt) in pair[0].targets for pt in config.event_types[self.pairtype]["subtypes"]):
                        trueddi = 1
                        truepcount += 1
                        strue += 1
                    else:
                        sfalse += 1
                    # true/total ratio
                    if train is True and trueddi == 0 and 1.0*strue/(strue+sfalse) < 0.01:
                        sfalse -= 1
                        skipped += 1
                        continue

                    else:

                        #pos_sentences.add(pair[0].sid)
                        #pos_sentences.add(pair[1].sid)
                        body = self.generatejSRE_line(tokens_text, pos, lemmas, ner)
                        examplelines.append(str(trueddi) + '\t' + pid + '.i' + '0\t' + body + '\n')
                        pcount += 1
            # print strue, sfalse, skipped
            #for il, l in enumerate(examplelines):

            logging.debug("writing {} lines to file...".format(len(examplelines)))
            with codecs.open(self.temp_dir + self.modelname + ".txt", 'a', "utf-8") as trainfile:
                for il, l in enumerate(examplelines):
                    # print sids[il], random.sample(pos_sentences, 1)
                    #if sids[il][0] in pos_sentences or  sids[il][1] in pos_sentences or not train:
                    trainfile.write(l)

                # logging.info("wrote " + temp_dir + savefile)
        logging.info("True/total relations:{}/{} ({})".format(truepcount, pcount, str(1.0*truepcount/(pcount+1))))

    def generatejSRE_line(self, pairtext, pos, lemmas, ner):
        candidates = [False,False]
        body = ''
        elements = []
        for it in range(len(pairtext)):
            #for it in range(len(pairtext)):
            if pairtext[it] == "#candidatea#":
                #print pairtext[i],
                tokentype = 'ENTITY'
                #tokentype = etypes[0]
                tokenlabel = 'A'
                candidates[0] = True
                #tokentext = "#candidate#"
                #tokentext = entitytext[0]
                tokentext = pairtext[it].lstrip()
                lemma = tokentext
            elif pairtext[it] == "#candidateb#":
                #print pairtext[i]
                tokentype = 'ENTITY'
                #tokentype = etypes[0]
                tokenlabel = 'T'
                #tokentext = "#candidate#"
                tokentext = pairtext[it].lstrip()
                #tokentext = entitytext[1]
                lemma = tokentext
                candidates[1] = True
            elif pairtext[it] == "#entity#":
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

            elements.append("&&".join([str(it), tokentext,
                              lemma,
                              pos[it],
                              tokentype, tokenlabel]))

        #logging.debug("%s\t%s\t%s", str(trueddi), pair, body)
        if not candidates[0]:
            logging.debug("missing first candidate on pair ")
            elements = ["0&&#candidate#&&#candidate#&&-None-&&ENTITY&&T"] + [str(n+1) + e[1:] for n, e in enumerate(elements)]
            # print pairtext
            # sys.exit()
        if not candidates[1]:
            logging.debug("missing second candidate on pair")
            elements.append(str(it+1) + "&&#candidate#&&#candidate#&&-None-&&ENTITY&&T")
            # print pairtext
            # sys.exit()
        body = " ".join(elements)
        return body

    def get_predictions(self, corpus):
        # real_pair_type = config.event_types[self.pairtype]["subtypes"][0]
        #pred_y = []
        with open(self.resultsfile, 'r') as resfile:
            pred = resfile.readlines()

        with codecs.open(self.examplesfile, 'r', 'utf-8') as trainfile:
            original = trainfile.readlines()

        if len(pred) != len(original):
            print "different number of predictions!"
            sys.exit()
        results = ResultsRE(self.resultsfile)
        temppreds = {}
        for i in range(len(pred)):
            original_tsv = original[i].split('\t')
            # logging.debug(original_tsv)
            pid = '.'.join(original_tsv[1].split('.')[:-1])
            #if pid not in pairs:
            #    print "pair not in pairs!"
            #    print pid
            #    print pairs
            #    sys.exit()
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
            if p == 1:
                did = '.'.join(pid.split(".")[:-1])
                if did not in results.document_pairs:
                    results.document_pairs[did] = Pairs()
                pair = corpus.documents[did].add_relation(self.pairs[pid][0], self.pairs[pid][1], self.pairtype, relation=True)
                # pair = corpus.documents[did].add_relation(self.pairs[pid][0], self.pairs[pid][1], real_pair_type, relation=True)
                #pair = self.get_pair(pid, corpus)
                results.pairs[pid] = pair
                results.document_pairs[did].add_pair(pair, "scikit")
                # logging.debug("{} - {} SLK: {}".format(pair.entities[0], pair.entities[1], p))
                #if pair not in temppreds:
                #    temppreds[pair] = []
                #temppreds[pair].append(p)
                results.pairs[pid].recognized_by["jsre"] = p
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
        results.corpus = corpus
        return results
