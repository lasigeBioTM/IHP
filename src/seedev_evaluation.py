import argparse
import logging
import os
import pickle

import time

import sys
from pycorenlp import StanfordCoreNLP

from classification.results import ResultsRE
from classification.rext.crfre import CrfSuiteRE
from classification.rext.jsrekernel import JSREKernel
from classification.rext.multir import MultiR
from classification.rext.rules import RuleClassifier
from classification.rext.scikitre import ScikitRE
from classification.rext.stanfordre import StanfordRE
from classification.rext.svmtk import SVMTKernel
from config import config
from evaluate import get_relations_results, get_gold_ann_set
from reader.seedev_corpus import SeeDevCorpus
from text.corpus import Corpus
from text.pair import Pairs


def write_seedev_results(results, path):
    if not os.path.isdir(path):
        os.makedirs(path)

    for did in results.document_pairs:
        with open(path + "/" + did + ".a2", 'w') as resfile:
            n = 1
            for pair in results.document_pairs[did].pairs:
                source_role = config.pair_types[pair.relation]["source_role"]
                target_role = config.pair_types[pair.relation]["target_role"]
                if pair.relation == "Exists_In_Genotype" and pair.entities[0].type == "Biological context":
                    source_role = "Element"
                elif pair.relation == "Is_Localized_In" and pair.entities[0].type in config.all_entity_groups["Dynamic_Process"]:
                    source_role = "Process"

                resfile.write("E{}\t{} {}:{} {}:{}\n".format(str(n), pair.relation, source_role, pair.entities[0].original_id,
                                                           target_role, pair.entities[1].original_id))
                n += 1

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("actions", default="classify",  help="Actions to be performed.")
    parser.add_argument("--goldstd", default="", dest="goldstd", nargs="+",
                      help="Gold standard to be used. Will override corpus, annotations",
                      choices=config.paths.keys())
    parser.add_argument("--submodels", default="", nargs='+', help="sub types of classifiers"),
    parser.add_argument("--models", dest="models", help="model destination path, without extension")
    parser.add_argument("--pairtype", dest="ptype", help="type of pairs to be considered", default="all")
    parser.add_argument("--doctype", dest="doctype", help="type of document to be considered", default="all")
    parser.add_argument("-o", "--output", "--format", dest="output",
                        nargs=2, help="format path; output formats: xml, html, tsv, text, chemdner.")
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
        # corpus = load_corpus(options.goldstd, corpus_path, corpus_format, corenlp_client)
        corpus = SeeDevCorpus(corpus_path)
        corpus.load_corpus(corenlp_client)
        corpus.save(config.paths[options.goldstd]["corpus"])
        if corpus_ann: #add annotation if it is not a test set
            corpus.load_annotations(corpus_ann, "all")
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
        # corpus.clear_annotations("all")
        corpus.load_annotations(corpus_ann, "all", options.ptype)
        # corpus.get_invalid_sentences()
        corpus.save(config.paths[options.goldstd]["corpus"])
    else:
        #corpus = SeeDevCorpus("corpus/" + "&".join(options.goldstd))
        corpus_path = config.paths[options.goldstd[0]]["corpus"]
        logging.info("loading corpus %s" % corpus_path)
        corpus = pickle.load(open(corpus_path, 'rb'))

        if options.actions == "add_sentences":
            corpus.add_more_sentences(options.models)

        elif options.actions == "train_relations":
            if options.ptype == "all":
                ptypes = config.pair_types.keys()
                # ptypes = config.event_types.keys()
            else:
                ptypes = [options.ptype]
            for p in ptypes:
                print p
                if options.kernel == "jsre":
                    model = JSREKernel(corpus, p, train=True)
                elif options.kernel == "svmtk":
                    model = SVMTKernel(corpus, p)
                elif options.kernel == "stanfordre":
                    model = StanfordRE(corpus, p)
                elif options.kernel == "multir":
                    model = MultiR(corpus, p)
                elif options.kernel == "scikit":
                    model = ScikitRE(corpus, p)
                elif options.kernel == "crf":
                    model = CrfSuiteRE(corpus, p)
                model.train()
        # testing

        elif options.actions == "test_relations":
            if options.ptype == "all":
                ptypes = config.pair_types.keys()
                # ptypes = config.event_types.keys()
                all_results = ResultsRE(options.output[1])
                all_results.corpus = corpus
                all_results.path = options.output[1]
            else:
                ptypes = [options.ptype]
            for p in ptypes:
                print p
                if options.kernel == "jsre":
                    model = JSREKernel(corpus, p, train=False)
                elif options.kernel == "svmtk":
                    model = SVMTKernel(corpus, p)
                elif options.kernel == "rules":
                    model = RuleClassifier(corpus, p)
                elif options.kernel == "stanfordre":
                    model = StanfordRE(corpus, p)
                elif options.kernel == "scikit":
                    model = ScikitRE(corpus, p)
                elif options.kernel == "crf":
                    model = CrfSuiteRE(corpus, p, test=True)
                model.load_classifier()
                model.test()
                results = model.get_predictions(corpus)
                # results.save(options.output[1] + "_" + p.lower() + ".pickle")
                # results.load_corpus(options.goldstd[0])
                results.path = options.output[1] + "_" + p.lower()
                goldset = get_gold_ann_set(config.paths[options.goldstd[0]]["format"], config.paths[options.goldstd[0]]["annotations"],
                                       "all", p, config.paths[options.goldstd[0]]["text"])
                get_relations_results(results, options.models, goldset[1],[], [])
                if options.ptype == "all":
                    for did in results.document_pairs:
                        if did not in all_results.document_pairs:
                            all_results.document_pairs[did] = Pairs(did=did)
                        all_results.document_pairs[did].pairs += results.document_pairs[did].pairs
            if options.ptype == "all":
                goldset = get_gold_ann_set(config.paths[options.goldstd[0]]["format"], config.paths[options.goldstd[0]]["annotations"],
                                       "all", "all", config.paths[options.goldstd[0]]["text"])
                get_relations_results(all_results, options.models, goldset[1],[], [])
                write_seedev_results(all_results, options.output[1])
        elif options.actions == "train_sentences": #and evaluate
            if options.ptype == "all":
                avg = [0,0,0]
                for p in config.pair_types:
                    print p
                    tps, fps, fns = corpus.train_sentence_classifier(p)
                    if tps == 0 and fns == 0:
                        precision, recall, fscore = 0, 1, 1
                    else:
                        precision = 1.0 * tps / (fps + tps)
                        recall = 1.0 * fns / (fns + tps)
                        fscore = 2.0 * precision * recall / (recall + precision)
                    print precision, recall, fscore
                    avg[0] += tps
                    avg[1] += fps
                    avg[2] += fns
                #print [a/len(config.pair_types) for a in avg]
                precision = 1.0 * avg[1] / (avg[0] + avg[1])
                recall = 1.0 * avg[2] / (avg[0] + avg[2])
                fscore = 2.0 * precision * recall / (recall + precision)
                print precision, recall, fscore
            else:
                res = corpus.train_sentence_classifier(options.ptype)
                print res
            corpus.save(config.paths[options.goldstd[0]]["corpus"])
        elif options.actions == "test_sentences": #and evaluate
            if options.ptype == "all":
                avg = [0,0,0]
                for p in config.pair_types:
                    print p
                    tps, fps, fns = corpus.test_sentence_classifier(p)
                if tps == 0 and fns == 0:
                    precision, recall, fscore = 0, 1, 1
                else:
                    precision = 1.0 * tps / (fps + tps)
                    recall = 1.0 * fns / (fns + tps)
                    fscore = 2.0 * precision * recall / (recall + precision)
                print precision, recall, fscore
                avg[0] += tps
                avg[1] += fps
                avg[2] += fns
            #print [a/len(config.pair_types) for a in avg]
            precision = 1.0 * avg[1] / (avg[0] + avg[1])
            recall = 1.0 * avg[2] / (avg[0] + avg[2])
            fscore = 2.0 * precision * recall / (recall + precision)
            print precision, recall, fscore
        else:
            res = corpus.test_sentence_classifier(options.ptype)
            print res
        corpus.save(config.paths[options.goldstd[0]]["corpus"])

    total_time = time.time() - start_time
    logging.info("Total time: %ss" % total_time)

if __name__ == "__main__":
    main()
