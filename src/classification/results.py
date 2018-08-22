import logging
import pickle
import os
import time
import argparse
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '../..'))
from text.corpus import Corpus
from config import config
from text.offset import Offset, perfect_overlap, contained_by, Offsets

SINGLE_TAG = "single"
START_TAG = "start"
END_TAG = "end"
MIDDLE_TAG = "middle"
OTHER_TAG = "other"

class ResultsRE(object):
    def __init__(self, name):
        self.pairs = {}
        self.name = name
        self.corpus = None
        self.document_pairs = {}

    def save(self, path):
        # no need to save the whole corpus, only the entities of each sentence are necessary
        # because the full corpus is already saved on a diferent pickle
        logging.info("Saving results to {}".format(path))
        reduced_corpus = {}
        npairs = 0
        for did in self.corpus.documents:
            self.document_pairs[did] = self.corpus.documents[did].pairs
            npairs += len(self.document_pairs[did].pairs)
            reduced_corpus[did] = {}
            for sentence in self.corpus.documents[did].sentences:
                reduced_corpus[did][sentence.sid] = sentence.entities
        self.corpus = reduced_corpus
        pickle.dump(self, open(path, "wb"))

    def load_corpus(self, goldstd):
        logging.info("loading corpus %s" % config.paths[goldstd]["corpus"])
        corpus = pickle.load(open(config.paths[goldstd]["corpus"]))

        for did in corpus.documents:
            for sentence in corpus.documents[did].sentences:
                sentence.entities = self.corpus[did][sentence.sid]
            corpus.documents[did].pairs = self.document_pairs[did]
                #for entity in sentence.entities.elist[options.models]:
                #    print entity.chebi_score,

        self.corpus = corpus

class ResultsNER(object):
    """Store a set of entities related to a corpus or input text """
    def __init__(self, name):
        self.entities = {}
        self.name = name
        self.corpus = Corpus(self.name)

    def get_ensemble_results(self, ensemble, corpus, model):
        """
            Go through every entity in corpus and if it was predicted true by the ensemble, save to entities,
            otherwise, delete it.
        """
        for did in corpus.documents:
            for sentence in corpus.documents[did].sentences:
                new_entities = []
                for entity in sentence.entities.elist[model]:
                    sentence_type = "A"
                    if sentence.sid.endswith("s0"):
                        sentence_type = "T"
                    id = (did, "{0}:{1}:{2}".format(sentence_type, entity.dstart, entity.dend), "1")
                    if id not in ensemble.ids:
                        logging.debug("this is new! {0}".format(entity))
                        continue
                    predicted_index = ensemble.ids.index(id)
                    #logging.info(predicted_index)
                    if ensemble.predicted[predicted_index][1] > 0.5:
                        self.entities[entity.eid] = entity
                        #logging.info("good entity: {}".format(entity.text.encode("utf8")))
                        new_entities.append(entity)
                    #else:
                    #    logging.info("bad entity: {}".format(entity.text.encode("utf8")))
                sentence.entities.elist[self.name] = new_entities
        self.corpus = corpus

    def save(self, path):
        # no need to save the whole corpus, only the entities of each sentence are necessary
        # because the full corpus is already saved on a diferent pickle
        logging.info("Saving results to {}".format(path))
        reduced_corpus = {}
        for did in self.corpus.documents:
            reduced_corpus[did] = {}
            for sentence in self.corpus.documents[did].sentences:
                reduced_corpus[did][sentence.sid] = sentence.entities
        self.corpus = reduced_corpus
        pickle.dump(self, open(path, "wb"))
    
    def save_chemdner(self):
        pass

    def load_corpus(self, goldstd):
        logging.info("loading corpus %s" % config.paths[goldstd]["corpus"])
        file = open(config.paths[goldstd]["corpus"], 'rb')
        corpus = pickle.load(file)
        for did in corpus.documents:
            for sentence in corpus.documents[did].sentences:
                sentence.entities = self.corpus[did][sentence.sid]
                #for entity in sentence.entities.elist[options.models]:
                #    print entity.chebi_score,

        self.corpus = corpus

    def combine_results(self, basemodel, name):
        # add another set of anotations to each sentence, ending in combined
        # each entity from this dataset should have a unique ID and a recognized_by attribute
        scores = 0
        total = 0
        for did in self.corpus.documents:
            #logging.debug(did)
            for sentence in self.corpus.documents[did].sentences:
                #logging.debug(sentence.sid)
                sentence.entities.combine_entities(basemodel, name)
                for e in sentence.entities.elist[name]:
                    total += 1
                    #logging.info("{} - {}".format(e.text, e.score))
                    if len(e.recognized_by) > 1:
                        scores += sum(e.score.values())/len(list(e.score.values()))
                    elif len == 1:
                        scores += list(e.score.values())[0]
                    #if e.score < 0.8:
                    #    logging.info("{0} score of {1}".format(e.text.encode("utf-8"),
                    #                                            e.score))
        if total > 0:
            logging.info("{0} entities average confidence of {1}".format(total, scores/total))


class ResultSetNER(object):
    """
    Organize and process a set a results from a TaggerCollection
    """
    def __init__(self, corpus, basepath):
        self.results = [] # list of ResultsNER
        self.corpus = corpus
        self.basepath = basepath

    def add_results(self, res):
        self.results.append(res)

    def combine_results(self):
        """
        Combine the results from multiple classifiers stored in self.results.
        Process these results, and generate a ResultsNER object
        :return: ResultsNER object of the combined results of the classifiers
        """
        final_results = ResultsNER(self.basepath)
        final_results.corpus = self.corpus
        return final_results


def combine_results(modelname, results, resultsname, etype, models):
    all_results = ResultsNER(resultsname)
    # first results are used as reference
    all_results.corpus = results[0].corpus
    for r in results:
        print(r.path)
        for did in r.corpus.documents:
            for sentence in r.corpus.documents[did].sentences:
                ref_sentence = all_results.corpus.documents[did].get_sentence(sentence.sid)
                if sentence.entities:
                    offsets = Offsets()
                    if modelname not in ref_sentence.entities.elist:
                        all_results.corpus.documents[did].get_sentence(sentence.sid).entities.elist[modelname] = []
                    for s in sentence.entities.elist:
                        # print s
                        if s in models:
                            # print s
                            for e in sentence.entities.elist[s]:
                                if e.type == etype:
                                    eid_offset = Offset(e.dstart, e.dend, text=e.text, sid=e.sid)
                                    exclude = [perfect_overlap]
                                    toadd, v, alt = offsets.add_offset(eid_offset, exclude_if=exclude)
                                    if toadd:
                                        # print "added:", r.path, s, e.text
                                        ref_sentence.entities.elist[modelname].append(e)
    return all_results

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("action", default="evaluate", help="Actions to be performed.")
    parser.add_argument("goldstd", default="chemdner_sample", help="Gold standard to be used.",
                        choices=list(config.paths.keys()))
    parser.add_argument("--corpus", dest="corpus",
                      default="data/chemdner_sample_abstracts.txt.pickle",
                      help="format path")
    parser.add_argument("--results", dest="results", help="Results object pickle.",  nargs='+')
    parser.add_argument("--models", dest="models", help="model destination path, without extension",  nargs='+')
    parser.add_argument("--finalmodel", dest="finalmodel", help="model destination path, without extension") #,  nargs='+')
    parser.add_argument("--ensemble", dest="ensemble", help="name/path of ensemble classifier", default="combined")
    parser.add_argument("--log", action="store", dest="loglevel", default="WARNING", help="Log level")
    parser.add_argument("-o", "--output", action="store", dest="output")
    parser.add_argument("--submodels", default="", nargs='+', help="sub types of classifiers"),
    parser.add_argument("--features", default=["chebi", "case", "number", "greek", "dashes", "commas", "length", "chemwords", "bow"],
                        nargs='+', help="aditional features for ensemble classifier")
    parser.add_argument("--doctype", dest="doctype", help="type of document to be considered", default="all")
    parser.add_argument("--entitytype", dest="etype", help="type of entities to be considered", default="all")
    parser.add_argument("--external", action="store_true", default=False, help="Run external evaluation script, depends on corpus type")
    options = parser.parse_args()

    numeric_level = getattr(logging, options.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % options.loglevel)
    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    logging_format = '%(asctime)s %(levelname)s %(filename)s:%(lineno)s:%(funcName)s %(message)s'
    logging.basicConfig(level=numeric_level, format=logging_format)
    logging.getLogger().setLevel(numeric_level)
    logging.info("Processing action {0} on {1}".format(options.action, options.goldstd))
    logging.info("loading results %s" % options.results + ".pickle")
    results_list = []
    for r in options.results:
        if os.path.exists(r + ".pickle"):
            results = pickle.load(open(r + ".pickle", 'rb'))
            results.path = r
            results.load_corpus(options.goldstd)
            results_list.append(results)
        else:
            print("results not found")

    if options.action == "combine":
        # add another set of annotations to each sentence, ending in combined
        # each entity from this dataset should have a unique ID and a recognized_by attribute
        logging.info("combining results...")
        #new_name = "_".join([m.split("/")[-1] for m in options.results])
        #print new_name
        results = combine_results(options.finalmodel, results_list, options.output, options.etype, options.models)
        results.save(options.output + ".pickle")

    """elif options.action in ("train_ensemble", "test_ensemble"):
        if "annotations" in config.paths[options.goldstd]:
            logging.info("loading gold standard %s" % config.paths[options.goldstd]["annotations"])
            goldset = get_gold_ann_set(config.paths[options.goldstd]["format"], config.paths[options.goldstd]["annotations"],
                                       options.etype,  config.paths[options.goldstd]["text"])
        else:
            goldset = None
        logging.info("using thresholds: chebi > {!s} ssm > {!s}".format(options.chebi, options.ssm))
        results.load_corpus(options.goldstd)
        results.path = options.results
        ths = {"chebi": options.chebi, "ssm": options.ssm}
        if "ensemble" in options.action:
            if len(options.submodels) > 1:
                submodels = []
                for s in options.submodels:
                    submodels += ['_'.join(options.models.split("_")[:-1]) + "_" + s + "_" + t for t in results.corpus.subtypes]
            else:
                submodels = ['_'.join(options.models.split("_")[:-1]) + "_" + t for t in results.corpus.subtypes]
            logging.info("using these features: {}".format(' '.join(submodels)))
        if options.action == "train_ensemble":
            ensemble = EnsembleNER(options.ensemble, goldset, options.models, types=submodels,
                                   features=options.features)
            ensemble.generate_data(results)
            ensemble.train()
            ensemble.save()
        if options.action == "test_ensemble":
            ensemble = EnsembleNER(options.ensemble, [], options.models, types=submodels,
                                   features=options.features)
            ensemble.load()
            ensemble.generate_data(results, supervisioned=False)
            ensemble.test()
            ensemble_results = ResultsNER(options.models + "_ensemble")
            # process the results
            ensemble_results.get_ensemble_results(ensemble, results.corpus, options.models)
            ensemble_results.path = options.results + "_ensemble"
            get_results(ensemble_results, options.models + "_ensemble", goldset, ths, options.rules)"""

    total_time = time.time() - start_time
    logging.info("Total time: %ss" % total_time)
if __name__ == "__main__":
    main()
