import itertools
import logging
import random
import time
import argparse
import pickle
import sys
from .classification.ner.crfsuitener import CrfSuiteModel
from .classification.ner.simpletagger import feature_extractors
from .classification.ner.stanfordner import StanfordNERModel
from .classification.results import ResultsNER
from .config import config
from .evaluate import get_gold_ann_set, get_results
# from postprocessing.chebi_resolution import add_chebi_mappings
# from postprocessing.ssm import add_ssm_score
from .text.corpus import Corpus


def run_crossvalidation(goldstd_list, corpus, model, cv, crf="stanford", entity_type="all", cvlog="cv.log"):
    logfile = open(cvlog, 'w')
    doclist = list(corpus.documents.keys())
    random.shuffle(doclist)
    size = int(len(doclist)/cv)
    sublists = chunks(doclist, size)
    logging.debug("Chunks:")
    logging.debug(sublists)
    p, r = [], []
    all_results = ResultsNER(model)
    all_results.path = model + "_results"
    for nlist in range(cv):
        testids, trainids = None, None
        testids = sublists[nlist]
        trainids = list(itertools.chain.from_iterable(sublists[:nlist]))
        trainids += list(itertools.chain.from_iterable(sublists[nlist+1:]))
        train_corpus, test_corpus = None, None
        print('CV{} - test set: {}; train set: {}'.format(nlist, len(testids), len(trainids)))
        train_corpus = Corpus(corpus.path + "_train", documents={did: corpus.documents[did] for did in trainids})
        test_corpus = Corpus(corpus.path + "_test", documents={did: corpus.documents[did] for did in testids})
        # logging.debug("train corpus docs: {}".format("\n".join(train_corpus.documents.keys())))
        #test_entities = len(test_corpus.get_all_entities("goldstandard"))
        #train_entities = len(train_corpus.get_all_entities("goldstandard"))
        #logging.info("test set entities: {}; train set entities: {}".format(test_entities, train_entities))
        basemodel = model + "_cv{}".format(nlist)
        logging.debug('CV{} - test set: {}; train set: {}'.format(nlist, len(test_corpus.documents), len(train_corpus.documents)))
        '''for d in train_corpus.documents:
            for s in train_corpus.documents[d].sentences:
                print len([t.tags.get("goldstandard") for t in s.tokens if t.tags.get("goldstandard") != "other"])
        sys.exit()'''
        # train
        logging.info('CV{} - TRAIN'.format(nlist))
        # train_model = StanfordNERModel(basemodel)
        train_model = None
        if crf == "stanford":
            train_model = StanfordNERModel(basemodel, entity_type)
        elif crf == "crfsuite":
            train_model = CrfSuiteModel(basemodel, entity_type)
        train_model.load_data(train_corpus, list(feature_extractors.keys()))
        train_model.train()

        # test
        logging.info('CV{} - TEST'.format(nlist))
        test_model = None
        if crf == "stanford":
            test_model = StanfordNERModel(basemodel, entity_type)
        elif crf == "crfsuite":
            test_model = CrfSuiteModel(basemodel, entity_type)
        test_model.load_tagger(port=9191+nlist)
        test_model.load_data(test_corpus, list(feature_extractors.keys()), mode="test")
        final_results = None
        final_results = test_model.test(test_corpus, port=9191+nlist)
        if crf == "stanford":
            test_model.kill_process()
        final_results.basepath = basemodel + "_results"
        final_results.path = basemodel

        all_results.entities.update(final_results.entities)
        all_results.corpus.documents.update(final_results.corpus.documents)
        # validate
        """if config.use_chebi:
            logging.info('CV{} - VALIDATE'.format(nlist))
            final_results = add_chebi_mappings(final_results, basemodel)
            final_results = add_ssm_score(final_results, basemodel)
            final_results.combine_results(basemodel, basemodel)"""

        # evaluate
        logging.info('CV{} - EVALUATE'.format(nlist))
        test_goldset = set()
        for gs in goldstd_list:
            goldset = get_gold_ann_set(config.paths[gs]["format"], config.paths[gs]["annotations"], entity_type,
                                       "pairtype", config.paths[gs]["text"] )
            for g in goldset[0]:
                if g[0] in testids:
                    test_goldset.add(g)
        precision, recall = get_results(final_results, basemodel, test_goldset, {}, [])
        # evaluation = run_chemdner_evaluation(config.paths[goldstd]["cem"], basemodel + "_results.txt", "-t")
        # values = evaluation.split("\n")[1].split('\t')
        p.append(precision)
        r.append(recall)
        # logging.info("precision: {} recall:{}".format(str(values[13]), str(values[14])))
    pavg = sum(p)/cv
    ravg = sum(r)/cv
    print("precision: average={} all={}".format(str(pavg), '|'.join([str(pp) for pp in p])))
    print("recall: average={}  all={}".format(str(ravg), '|'.join([str(rr) for rr in r])))
    all_goldset = set()
    for gs in goldstd_list:
        goldset = get_gold_ann_set(config.paths[gs]["format"], config.paths[gs]["annotations"], entity_type,
                                       config.paths[gs]["text"] )
        for g in goldset:
            all_goldset.add(g)
    get_results(all_results, model, all_goldset, {}, [])


def chunks(l, n):
    """ return list of n sized sublists of l
    """
    subs = []
    for i in range(0, len(l), n):
        subs.append(l[i:i+n])
    return subs

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--goldstd", default="", dest="goldstd", nargs="+",
                      help="Gold standard to be used. Will override corpus, annotations",
                      choices=list(config.paths.keys()))
    parser.add_argument("--submodels", default="", nargs='+', help="sub types of classifiers"),
    parser.add_argument("--corpus", dest="corpus", nargs=2,
                      default=["chemdner", "CHEMDNER/CHEMDNER_SAMPLE_JUNE25/chemdner_sample_abstracts.txt"],
                      help="format path")
    parser.add_argument("--annotations", dest="annotations")
    parser.add_argument("--tag", dest="tag", default="0", help="Tag to identify the text.")
    parser.add_argument("--cv", dest="cv", default=5, help="Number of folds.", type=int)
    parser.add_argument("--models", dest="models", help="model destination path, without extension")
    parser.add_argument("--entitytype", dest="etype", help="type of entities to be considered", default="all")
    parser.add_argument("--doctype", dest="doctype", help="type of document to be considered", default="all")
    parser.add_argument("-o", "--output", "--format", dest="output",
                        nargs=2, help="format path; output formats: xml, html, tsv, text, chemdner.")
    parser.add_argument("--crf", dest="crf", help="CRF implementation", default="stanford",
                        choices=["stanford", "crfsuite"])
    parser.add_argument("--log", action="store", dest="loglevel", default="WARNING", help="Log level")
    parser.add_argument("--kernel", action="store", dest="kernel", default="svmtk", help="Kernel for relation extraction")
    parser.add_argument("--pairtype1", action="store", dest="pairtype1")
    parser.add_argument("--pairtype2", action="store", dest="pairtype2")
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
    logging.info("Crossvalidation on {0}".format(options.goldstd))

    # set configuration variables based on the goldstd option if the corpus has a gold standard,
    # or on corpus and annotation options
    # pre-processing options
    corpus_name = "&".join(options.goldstd)
    corpus = Corpus("corpus/" + corpus_name)
    for g in options.goldstd:
        corpus_path = config.paths[g]["corpus"]
        logging.info("loading corpus %s" % corpus_path)
        this_corpus = pickle.load(open(corpus_path, 'rb'))
        corpus.documents.update(this_corpus.documents)
    run_crossvalidation(options.goldstd, corpus, options.models, options.cv, options.crf, options.etype)

    total_time = time.time() - start_time
    logging.info("Total time: %ss" % total_time)

if __name__ == "__main__":
    main()