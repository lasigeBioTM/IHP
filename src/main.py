#!/usr/bin/env python
from __future__ import division, unicode_literals

import argparse
import cPickle as pickle
import codecs
import logging
import sys
import time

from pycorenlp import StanfordCoreNLP

from classification.ner.crfsuitener import CrfSuiteModel
from classification.ner.matcher import MatcherModel
from classification.ner.mirna_matcher import MirnaMatcher
from classification.ner.simpletagger import BiasModel, feature_extractors
from classification.ner.stanfordner import StanfordNERModel
from classification.ner.taggercollection import TaggerCollection
from classification.results import ResultsNER, ResultSetNER
from classification.rext.crfre import CrfSuiteRE
from classification.rext.jsrekernel import JSREKernel
#from classification.rext.multir import MultiR
from classification.rext.rules import RuleClassifier
from classification.rext.scikitre import ScikitRE
from classification.rext.stanfordre import StanfordRE
from classification.rext.svmtk import SVMTKernel
from config import config
from reader.chebi_corpus import ChebiCorpus
from reader.chemdner_corpus import ChemdnerCorpus
from reader.ddi_corpus import DDICorpus
from reader.genia_corpus import GeniaCorpus
from reader.gpro_corpus import GproCorpus
from reader.jnlpba_corpus import JNLPBACorpus
from reader.mirna_corpus import MirnaCorpus
from reader.mirtext_corpus import MirtexCorpus
from reader.pubmed_corpus import PubmedCorpus
from reader.tempEval_corpus import TempEvalCorpus
from reader.transmir_corpus import TransmirCorpus
from reader.hpo_corpus import HPOCorpus ######### 
from reader.test_suite import SuiteCorpus #######
from text.corpus import Corpus

if config.use_chebi:
    pass

def load_corpus(goldstd, corpus_path, corpus_format, corenlp_client):
    if corpus_format == "chemdner":
        corpus = ChemdnerCorpus(corpus_path)
        corpus.load_corpus(corenlp_client)
    elif corpus_format == "gpro":
        corpus = GproCorpus(corpus_path)
        corpus.load_corpus(None)
    elif corpus_format == "ddi":
        corpus = DDICorpus(corpus_path)
        corpus.load_corpus(corenlp_client)
        # since the path of this corpus is a directory, add the reference to save this corpus
        corpus.path += goldstd + ".txt"
    elif corpus_format == "chebi":
        corpus = ChebiCorpus(corpus_path)
        corpus.load_corpus(corenlp_client)
        # since the path of this corpus is a directory, add the reference to save this corpus
        corpus.path += goldstd + ".txt"
    elif corpus_format == "tempeval":
        corpus = TempEvalCorpus(corpus_path)
        corpus.load_corpus(corenlp_client)
    elif corpus_format == "pubmed":
        # corenlpserver = ""
        with open(corpus_path, 'r') as f:
            pmids = [line.strip() for line in f if line.strip()]
        corpus = PubmedCorpus(corpus_path, pmids)
        corpus.load_corpus(corenlp_client)
    elif corpus_format == "genia":
        corpus = GeniaCorpus(corpus_path)
        corpus.load_corpus(corenlp_client)
    elif corpus_format == "ddi-mirna":
        corpus = MirnaCorpus(corpus_path)
        corpus.load_corpus(corenlp_client)
    elif corpus_format == "mirtex":
        corpus = MirtexCorpus(corpus_path)
        corpus.load_corpus(corenlp_client)
        # corpus.path = ".".join(config.paths[options.goldstd]["corpus"].split(".")[:-1])
    elif corpus_format == "transmir":
        corpus = TransmirCorpus(corpus_path)
        corpus.load_corpus(corenlp_client)
    elif corpus_format == "jnlpba":
        corpus = JNLPBACorpus(corpus_path)
        corpus.load_corpus(corenlp_client)
    elif corpus_format == "hpo":            ##########
        corpus = HPOCorpus(corpus_path)     ##########
        corpus.load_corpus(corenlp_client)  ##########
    elif corpus_format == "tsuite":            ##########
        corpus = SuiteCorpus(corpus_path)     ##########
        corpus.load_corpus(corenlp_client)  ##########
    return corpus

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("actions", default="classify",  help="Actions to be performed.",
                      choices=["load_corpus", "annotate", "classify", "write_results", "write_goldstandard",
                               "train", "test", "train_multiple", "test_multiple", "train_matcher", "test_matcher",
                               "crossvalidation", "train_relations", "test_relations"])
    parser.add_argument("--goldstd", default="", dest="goldstd", nargs="+",
                      help="Gold standard to be used. Will override corpus, annotations",
                      choices=config.paths.keys())
    parser.add_argument("--submodels", default="", nargs='+', help="sub types of classifiers"),
    parser.add_argument("-i", "--input", dest="input", action="store",
                      default='''Administration of a higher dose of indinavir should be \\
considered when coadministering with megestrol acetate.''',
                      help="Text to classify.")
    parser.add_argument("--corpus", dest="corpus", nargs=2,
                      default=["chemdner", "CHEMDNER/CHEMDNER_SAMPLE_JUNE25/chemdner_sample_abstracts.txt"],
                      help="format path")
    parser.add_argument("--annotations", dest="annotations")
    parser.add_argument("--tag", dest="tag", default="0", help="Tag to identify the text.")
    parser.add_argument("--models", dest="models", help="model destination path, without extension")
    parser.add_argument("--entitytype", dest="etype", help="type of entities to be considered", default="all")
    parser.add_argument("--pairtype", dest="ptype", help="type of pairs to be considered", default="all")
    parser.add_argument("--doctype", dest="doctype", help="type of document to be considered", default="all")
    parser.add_argument("--annotated", action="store_true", default=False, dest="annotated",
                      help="True if the input has <entity> tags.")
    parser.add_argument("-o", "--output", "--format", dest="output",
                        nargs=2, help="format path; output formats: xml, html, tsv, text, chemdner.")
    parser.add_argument("--crf", dest="crf", help="CRF implementation", default="stanford",
                        choices=["stanford", "crfsuite"])
    parser.add_argument("--log", action="store", dest="loglevel", default="WARNING", help="Log level")
    parser.add_argument("--kernel", action="store", dest="kernel", default="svmtk", help="Kernel for relation extraction")
    options = parser.parse_args()

    # set logger
    numeric_level = getattr(logging, options.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % options.loglevel)
    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    logging_format = '%(asctime)s %(levelname)s %(filename)s:%(lineno)s:%(funcName)s %(message)s'
    logging.basicConfig(level=numeric_level, format=logging_format)
    logging.getLogger().setLevel(numeric_level)
    logging.getLogger("requests.packages").setLevel(30)
    logging.info("Processing action {0} on {1}".format(options.actions, options.goldstd))

    # set configuration variables based on the goldstd option if the corpus has a gold standard,
    # or on corpus and annotation options
    # pre-processing options
    if options.actions == "load_corpus":
        if len(options.goldstd) > 1:
            print "load only one corpus each time"
            sys.exit()
        options.goldstd = options.goldstd[0]
        corpus_format = config.paths[options.goldstd]["format"]
        corpus_path = config.paths[options.goldstd]["text"]
        corpus_ann = config.paths[options.goldstd]["annotations"]

        corenlp_client = StanfordCoreNLP('http://localhost:9000')
        corpus = load_corpus(options.goldstd, corpus_path, corpus_format, corenlp_client)
        corpus.save(config.paths[options.goldstd]["corpus"])
        if corpus_ann: #add annotation if it is not a test set
            corpus.load_annotations(corpus_ann, options.etype, options.ptype)
            corpus.save(config.paths[options.goldstd]["corpus"])

    elif options.actions == "annotate": # rext-add annotation to corpus
        if len(options.goldstd) > 1:
            print "load only one corpus each time"
            sys.exit()
        options.goldstd = options.goldstd[0]
        corpus_path = config.paths[options.goldstd]["corpus"]
        corpus_ann = config.paths[options.goldstd]["annotations"]
        logging.info("loading corpus %s" % corpus_path)
        corpus = pickle.load(open(corpus_path, 'rb'))
        logging.debug("loading annotations...")
        corpus.clear_annotations(options.etype)
        corpus.load_annotations(corpus_ann, options.etype, options.ptype)
        # corpus.get_invalid_sentences()
        corpus.save(config.paths[options.goldstd]["corpus"])
    else:
        corpus = Corpus("corpus/" + "&".join(options.goldstd))
        for g in options.goldstd:
            corpus_path = config.paths[g]["corpus"]
            logging.info("loading corpus %s" % corpus_path)
            this_corpus = pickle.load(open(corpus_path, 'rb'))
            corpus.documents.update(this_corpus.documents)
        if options.actions == "write_goldstandard":
            model = BiasModel(options.output[1])
            model.load_data(corpus, [])
            results = model.test()
            #results = ResultsNER(options.output[1])
            #results.get_ner_results(corpus, model)
            results.save(options.output[1] + ".pickle")
            #logging.info("saved gold standard results to " + options.output[1] + ".txt")

        # training
        elif options.actions == "train":
            if options.crf == "stanford":
                model = StanfordNERModel(options.models, options.etype)
            elif options.crf == "crfsuite":
                model = CrfSuiteModel(options.models, options.etype)
            model.load_data(corpus, feature_extractors.keys(), options.etype)
            model.train()
        elif options.actions == "train_matcher": # Train a simple classifier based on string matching
            model = MatcherModel(options.models)
            model.train(corpus)
            # TODO: term list option
            #model.train("TermList.txt")
        elif options.actions == "train_multiple": # Train one classifier for each type of entity in this corpus
            # logging.info(corpus.subtypes)
            models = TaggerCollection(basepath=options.models, corpus=corpus, subtypes=corpus.subtypes)
            models.train_types()
        elif options.actions == "train_relations":
            if options.kernel == "jsre":
                model = JSREKernel(corpus, options.ptype)
            elif options.kernel == "svmtk":
                model = SVMTKernel(corpus, options.ptype)
            elif options.kernel == "stanfordre":
                model = StanfordRE(corpus, options.ptype)
            elif options.kernel == "multir":
                model = MultiR(corpus, options.ptype)
            elif options.kernel == "scikit":
                model = ScikitRE(corpus, options.ptype)
            elif options.kernel == "crf":
                model = CrfSuiteRE(corpus, options.ptype)
            model.train()
        # testing
        elif options.actions == "test":
            base_port = 9191
            if len(options.submodels) > 1:
                allresults = ResultSetNER(corpus, options.output[1])
                for i, submodel in enumerate(options.submodels):
                    model = StanfordNERModel(options.models + "_" + submodel)
                    model.load_tagger(base_port + i)
                    # load data into the model format
                    model.load_data(corpus, feature_extractors.keys(), mode="test")
                    # run the classifier on the data
                    results = model.test(corpus, port=base_port + i)
                    allresults.add_results(results)
                    model.kill_process()
                # save the results to an object that can be read again, and log files to debug
                final_results = allresults.combine_results()
            else:
                if options.crf == "stanford":
                    model = StanfordNERModel(options.models, options.etype)
                elif options.crf == "crfsuite":
                    model = CrfSuiteModel(options.models, options.etype)
                model.load_tagger()
                model.load_data(corpus, feature_extractors.keys(), mode="test")
                final_results = model.test(corpus)
            #with codecs.open(options.output[1] + ".txt", 'w', 'utf-8') as outfile:
            #    lines = final_results.corpus.write_chemdner_results(options.models, outfile)
            #final_results.lines = lines
            final_results.save(options.output[1] + ".pickle")
        elif options.actions == "test_matcher":
            if "mirna" in options.models:
                model = MirnaMatcher(options.models)
            else:
                model = MatcherModel(options.models)
            results = ResultsNER(options.models)
            results.corpus, results.entities = model.test(corpus)
            allentities = set()
            for e in results.entities:
                allentities.add(results.entities[e].text)
            with codecs.open(options.output[1] + ".txt", 'w', 'utf-8') as outfile:
                outfile.write('\n'.join(allentities))

            results.save(options.output[1] + ".pickle")
        elif options.actions == "test_multiple":
            logging.info("testing with multiple classifiers... {}".format(' '.join(options.submodels)))
            allresults = ResultSetNER(corpus, options.output[1])
            if len(options.submodels) < 2:
                models = TaggerCollection(basepath=options.models)
                models.load_models()
                results = models.test_types(corpus)
                final_results = results.combine_results()
            else:
                base_port = 9191
                for submodel in options.submodels:
                    models = TaggerCollection(basepath=options.models + "_" + submodel, baseport = base_port)
                    models.load_models()
                    results = models.test_types(corpus)
                    logging.info("combining results...")
                    submodel_results = results.combine_results()
                    allresults.add_results(submodel_results)
                    base_port += len(models.models)
                final_results = allresults.combine_results()
            logging.info("saving results...")
            final_results.save(options.output[1] + ".pickle")
        elif options.actions == "test_relations":
            if options.kernel == "jsre":
                model = JSREKernel(corpus, options.ptype, train=False)
            elif options.kernel == "svmtk":
                model = SVMTKernel(corpus, options.ptype)
            elif options.kernel == "rules":
                model = RuleClassifier(corpus, options.ptype)
            elif options.kernel == "stanfordre":
                model = StanfordRE(corpus, options.ptype)
            elif options.kernel == "scikit":
                model = ScikitRE(corpus, options.ptype)
            elif options.kernel == "crf":
                model = CrfSuiteRE(corpus, options.ptype, test=True)
            model.load_classifier()
            model.test()
            results = model.get_predictions(corpus)
            results.save(options.output[1] + ".pickle")

    total_time = time.time() - start_time
    logging.info("Total time: %ss" % total_time)

if __name__ == "__main__":
    main()
