import argparse
import logging
import os
import pickle
import time

import sys

from config import config

def add_chebi_mappings(results, path, source, save=True):
    """
    Go through each identified entity and add ChEBI mapping
    :param results: ResultsNER object
    :param path: Path where the results should be saved
    :param source: Base model path
    :param save: Save results to file
    :return: ResultsNER object
    """
    mapped = 0
    not_mapped = 0
    total_score = 0
    for idid, did in enumerate(results.corpus):
        logging.info("{}/{}".format(idid, len(results.corpus)))
        for sid in results.corpus[did]:
            for s in results.corpus[did][sid].elist:
                if s.startswith(source):
                    #if s != source:
                    #    logging.info("processing %s" % s)
                    for entity in results.corpus[did][sid].elist[s]:
                        chebi_info = chebi_resolution.find_chebi_term3(entity.text.encode("utf-8"))
                        entity.chebi_id = chebi_info[0]
                        entity.chebi_name = chebi_info[1]
                        entity.chebi_score = chebi_info[2]
                        # TODO: check for errors (FP and FN)
                        if chebi_info[2] == 0:
                            #logging.info("nothing for %s" % entity.text)
                            not_mapped += 1
                        else:
                            #logging.info("%s => %s %s" % (entity.text, chebi_info[1], chebi_info[2]))
                            mapped += 1
                            total_score += chebi_info[2]
    if mapped == 0:
        percentmapped = 0
    else:
        percentmapped = total_score/mapped
    logging.info("{0} mapped, {1} not mapped, average score: {2}".format(mapped, not_mapped, percentmapped))
    if save:
        logging.info("saving results to %s" % path)
        pickle.dump(results, open(path, "wb"))
    return results


def add_ssm_score(results, path, source, measure, ontology, save=True):
    """
    Add SSM score and info to each entity identified by source
    :param results: ResultsNER object
    :param path: Results path
    :param source: Base model path
    :param measure: Semantic similarity measure to be used
    :param ontology: Ontology to be used
    :param save: Save results
    :return: ResultsNER object
    """
    # calculate similarity at the level of the document instead of sentence
    total = 0
    scores = 0
    for did in results.corpus:
        entities = {} # get all the entities from this document
        for sid in results.corpus[did]:
            for s in results.corpus[did][sid].elist:
                if s.startswith(source):
                    if s not in entities:
                        entities[s] = []
                    entities[s] += results.corpus[did][sid].elist[s]

        for s in entities: # get SS within the entities of this document
            entities_ssm = get_ssm(entities[s], measure, ontology)
            scores += sum([e.ssm_score for e in entities_ssm])
            for e in entities_ssm: # add SSM info to results
                total += 1
                for e2 in results.corpus[did][e.sid].elist[s]:
                    if e2.eid == e.eid:
                        e2.ssm_score = e.ssm_score
                        e2.ssm_best_text = e.ssm_best_text
                        e2.ssm_best_ID = e.ssm_best_ID
                        e2.ssm_best_name = e.ssm_best_name
                        e2.ssm = measure
                    #for entity in results.corpus[did][sid].elist[s]:
                    #    logging.info("%s %s %s %s" % (entity.text, entity.chebi_name, entity.ssm_score,
                    #                                  entity.ssm_chebi_name))
    if total == 0:
        averagessm = 0
    else:
        averagessm = scores/total
    logging.info("average {1} ssm score: {0}".format(averagessm, measure))

    if save:
        logging.info("saving results to %s" % path)
        pickle.dump(results, open(path, "wb"))
    return results


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("action", default="map",
                      help="Actions to be performed.")
    parser.add_argument("goldstd", default="chemdner_sample",
                      help="Gold standard to be used.",
                      choices=config.paths.keys())
    parser.add_argument("--corpus", dest="corpus",
                      default="data/chemdner_sample_abstracts.txt.pickle",
                      help="format path")
    parser.add_argument("--results", dest="results", help="Results object pickle.")
    parser.add_argument("--models", dest="models", help="model destination path, without extension", default="combined")
    parser.add_argument("--ensemble", dest="ensemble", help="name/path of ensemble classifier", default="combined")
    parser.add_argument("--chebi", dest="chebi", help="Chebi mapping threshold.", default=0, type=float)
    parser.add_argument("--ssm", dest="ssm", help="SSM threshold.", default=0, type=float)
    parser.add_argument("--measure", dest="measure", help="semantic similarity measure", default="simui")
    parser.add_argument("--log", action="store", dest="loglevel", default="WARNING", help="Log level")
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
    if os.path.exists(options.results + ".pickle"):
        results = pickle.load(open(options.results + ".pickle", 'rb'))
        results.path = options.results
    else:
        print "results not found"
        results = None

    if options.action == "chebi":
        if not config.use_chebi:
            print "If you want to use ChEBI, please re-run config.py and set use_chebi to true"
            sys.exit()
        add_chebi_mappings(results, options.results + ".pickle", options.models)
    # if options.action == "go":
    #    add_go_mappings(results, options.results + ".pickle", options.models)
    elif options.action == "mirna":
        pass
    elif options.action == "ssm":
        if options.measure.endswith("go"):
            ontology = "go"
        else:
            ontology = "chebi"
        add_ssm_score(results, options.results + ".pickle", options.models, options.measure, ontology)

if __name__ == "__main__":
    main()