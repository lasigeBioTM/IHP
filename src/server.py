import argparse
import time
import json

__author__ = 'Andre'
import bottle
from pycorenlp import StanfordCoreNLP
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import logging
import codecs
import cPickle as pickle
import random
import string

from text.document import Document
from text.corpus import Corpus
from classification.ner.taggercollection import TaggerCollection
from classification.ner.simpletagger import SimpleTaggerModel, feature_extractors
from postprocessing.ensemble_ner import EnsembleNER
from reader import pubmed
from text.pair import Pair
from config import config
from postprocessing.chebi_resolution import add_chebi_mappings
from postprocessing.ssm import add_ssm_score



class IICEServer(object):

    def __init__(self, basemodel, ensemble_model, submodels):
        self.corenlp = StanfordCoreNLP(config.corenlp_dir)
        self.basemodel = basemodel
        self.ensemble_model = ensemble_model
        self.subtypes = submodels
        self.models = TaggerCollection(basepath=self.basemodel)
        self.models.load_models()
        self.ensemble = EnsembleNER(self.ensemble_model, None, self.basemodel + "_combined", types=self.subtypes,
                                   features=[])
        self.ensemble.load()

    def hello(self):
        return "Hello World!"

    def process_pubmed(self, pmid):
        title, text = pubmed.get_pubmed_abs(pmid)

    def id_generator(self, size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    def process_multiple(self):
        bottle.response.content_type = "application/json"
        data = bottle.request.json
        text = data["text"]
        format = data["format"]
        test_corpus = self.generate_corpus(text)
        multiple_results = self.models.test_types(test_corpus)
        final_results = multiple_results.combine_results()
        final_results = add_chebi_mappings(final_results, self.basemodel)
        final_results = add_ssm_score(final_results, self.basemodel)
        final_results.combine_results(self.basemodel, self.basemodel + "_combined")

        # self.ensemble.generate_data(final_results, supervisioned=False)
        #self.ensemble.test()
        # ensemble_results = ResultsNER(self.basemodel + "_combined_ensemble")
        # ensemble_results.get_ensemble_results(self.ensemble, final_results.corpus, self.basemodel + "_combined")
        #output = get_output(final_results, basemodel + "_combined")
        results_id = self.id_generator()
        #output = self.get_output(ensemble_results, self.basemodel + "_combined_ensemble", format, id=results_id)
        output = self.get_output(final_results, self.basemodel + "_combined", format=format, results_id=results_id)
        #self.models.load_models()
        self.clean_up()
        # save corpus to pickel and add ID to the output as corpusfile
        pickle.dump(final_results.corpus, open("temp/{}.pickle".format(results_id), 'w'))
        return output

    def clean_up(self):

        for m in self.models.models:
            self.models.models[m].reset()
        self.models.basemodel.reset()

    def process(self, text="", modeltype="all"):
        test_corpus = self.generate_corpus(text)
        model = SimpleTaggerModel("models/chemdner_train_f13_lbfgs_" + modeltype)
        model.load_tagger()
        # load data into the model format
        model.load_data(test_corpus, feature_extractors.keys())
        # run the classifier on the data
        results = model.test(stats=False)
        #results = ResultsNER("models/chemdner_train_f13_lbfgs_" + modeltype)
        # process the results
        #results.get_ner_results(test_corpus, model)
        output = self.get_output(results, "models/chemdner_train_f13_lbfgs_" + modeltype)
        return output

    def get_relations(self):
        """
        Process the results dictionary, identify relations
        :return: results dictionary with relations
        """
        data = bottle.request.json
        # logging.debug(str(data))
        if "corpusfile" in data:
            corpus = pickle.load(open("temp/{}.pickle".format(data["corpusfile"])))
            logging.info("loaded corpus {}".format(data["corpusfile"]))
        else:
            # create corpus
            corpus = None
            pass
        did = "d0"
        #for sentence in data["abstract"]["sentences"]:
        for sentence in corpus.documents["d0"].sentences[1:]:
            sentence_pairs = []
            sentence_entities = sentence.entities.elist[self.basemodel + "_combined"]
            # logging.info("sentence entities:" + str(sentence_entities))
            sid = sentence.sid
            for i1, e1 in enumerate(sentence_entities):
                logging.info("sentence entities:" + str(e1))
                if i1 < len(sentence_entities)-1:
                    for i2, e2 in enumerate(sentence_entities[i1+1:]):
                        logging.info("sentence entities:" + str(e2))
                        pid = sentence.sid + ".p{}".format(len(sentence_pairs))
                        newpair = Pair(entities=[e1, e2], sid=sid, pid=pid, did=did)
                        sentence_pairs.append(newpair)
                        sentence.pairs.pairs[pid] = newpair
            logging.info(str(sentence_pairs))
            if len(sentence_pairs) > 0:
                corpus.documents[did].get_sentence(sid).test_relations(sentence_pairs, self.basemodel + "_combined")


        return data

    def generate_corpus(self, text):
        """
        Create a corpus object from the input text.
        :param text:
        :return:
        """
        test_corpus = Corpus("")
        newdoc = Document(text, process=False, did="d0", title="Test document")
        newdoc.sentence_tokenize("biomedical")
        newdoc.process_document(self.corenlp, "biomedical")
        test_corpus.documents["d0"] = newdoc
        return test_corpus

    def get_output(self, results, model_name, format="bioc", results_id=None):
        if format == "bioc":
            a = ET.Element('collection')
            bioc = results.corpus.documents["d0"].write_bioc_results(a, model_name)
            rough_string = ET.tostring(a, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            output = reparsed.toprettyxml(indent="\t")
        elif format == "chemdner":
             with codecs.open("/dev/null", 'w', 'utf-8') as outfile:
                lines = results.corpus.write_chemdner_results(model_name, outfile)
                output = ""
                for l in lines:
                    output += ' '.join(l) + " "
        else: # default should be json
            results_dic = results.corpus.documents["d0"].get_dic(model_name)
            results_dic["corpusfile"] = results_id
            output = json.dumps(results_dic)
        return output


def main():
    starttime = time.time()
    #parser = argparse.ArgumentParser(description='')
    #parser.add_argument("action", choices=["start", "stop"])
    #parser.add_argument("--basemodel", dest="model", help="base model path")
    #parser.add_argument("-t", "--test", action="store_true",
    #                help="run a test instead of the server")
    #parser.add_argument("--log", action="store", dest="loglevel", default="WARNING", help="Log level")
    #options = parser.parse_args()
    numeric_level = getattr(logging, "DEBUG", None)
    #if not isinstance(numeric_level, int):
    #    raise ValueError('Invalid log level: %s' % options.loglevel)

    #while len(logging.root.handlers) > 0:
    #    logging.root.removeHandler(logging.root.handlers[-1])
    logging_format = '%(asctime)s %(levelname)s %(filename)s:%(lineno)s:%(funcName)s %(message)s'
    logging.basicConfig(level=numeric_level, format=logging_format)
    logging.getLogger().setLevel(numeric_level)

    base_model = "models/chemdner_train"
    ensemble_model = "models/ensemble_ner/train_on_dev"
    #submodels = [base_model + "_" + t for t in ChemdnerCorpus.chemdner_types]
    if base_model.split("/")[-1].startswith("chemdner+ddi"):
        subtypes = ["drug", "group", "brand", "drug_n"] + ["IDENTIFIER", "MULTIPLE",
                                                                 "FAMILY", "FORMULA", "SYSTEMATIC",
                                                                 "ABBREVIATION", "TRIVIAL"] + ["chemdner", "ddi"]
    elif base_model.split("/")[-1].startswith("ddi"):
        subtypes = ["drug", "group", "brand", "drug_n", "all"]
    elif base_model.split("/")[-1].startswith("chemdner"):
        subtypes = ["IDENTIFIER", "MULTIPLE", "FAMILY", "FORMULA", "SYSTEMATIC", "ABBREVIATION", "TRIVIAL", "all"]
    submodels = ['_'.join(base_model.split("_")[:]) + "_" + t for t in subtypes[:-1]] #ignore the "all"
    logging.debug("Initializing the server...")
    server = IICEServer(base_model, ensemble_model, submodels)
    logging.debug("done.")
    load_time = time.time() - starttime
    starttime = time.time()
    '''if options.test:
        text = "Primary Leydig cells obtained from bank vole testes and the established tumor Leydig cell line (MA-10) have been used to explore the effects of 4-tert-octylphenol (OP). Leydig cells were treated with two concentrations of OP (10(-4)M, 10(-8)M) alone or concomitantly with anti-estrogen ICI 182,780 (1M). In OP-treated bank vole Leydig cells, inhomogeneous staining of estrogen receptor (ER) within cell nuclei was found, whereas it was of various intensity among MA-10 Leydig cells. The expression of ER mRNA and protein decreased in both primary and immortalized Leydig cells independently of OP dose. ICI partially reversed these effects at mRNA level while at protein level abrogation was found only in vole cells. Dissimilar action of OP on cAMP and androgen production was also observed. This study provides further evidence that OP shows estrogenic properties acting on Leydig cells. However, its effect is diverse depending on the cellular origin. "
        logging.debug("test with this text: {}".format(text))
        output = server.process_multiple(text)
        print output
        process_time = time.time() - starttime
        # second document
        text = "Azole class of compounds are well known for their excellent therapeutic properties. Present paper describes about the synthesis of three series of new 1,2,4-triazole and benzoxazole derivatives containing substituted pyrazole moiety (11a-d, 12a-d and 13a-d). The newly synthesized compounds were characterized by spectral studies and also by C, H, N analyses. All the synthesized compounds were screened for their analgesic activity by the tail flick method. The antimicrobial activity of the new derivatives was also performed by Minimum Inhibitory Concentration (MIC) by the serial dilution method. The results revealed that the compound 11c having 2,5-dichlorothiophene substituent on pyrazole moiety and a triazole ring showed significant analgesic and antimicrobial activity."
        logging.debug("test with this text: {}".format(text))
        output = server.process_multiple(text)
        print output
        logging.info("loading time: {}; process time: {}".format(load_time, process_time))
    else:'''
    bottle.route("/hello")(server.hello)
    bottle.route("/iice/chemical/entities", method='POST')(server.process_multiple)
    #bottle.route("/iice/chemical/<text>/<modeltype>", method='POST')(server.process)
    bottle.route("/iice/chemical/interactions", method='POST')(server.get_relations)
    #daemon_run(host='10.10.4.63', port=8080, logfile="server.log", pidfile="server.pid")
    bottle.run(host=config.host_ip, port=8080, DEBUG=True)
if __name__ == "__main__":
    main()