import logging
import multiprocessing

from classification.ner.simpletagger import feature_extractors
from classification.results import ResultSetNER
from classification.ner.stanfordner import StanfordNERModel

chemdnerModels = "bc_systematic bc_formula bc_trivial bc_abbreviation bc_family"


class TaggerCollection(object):
    """
    Collection of tagger classifiers used to train and test specific subtype models
    """
    CHEMDNER_TYPES =  ["IDENTIFIER", "MULTIPLE", "FAMILY", "FORMULA", "SYSTEMATIC", "ABBREVIATION", "TRIVIAL"]
    GPRO_TYPES = ["NESTED", "IDENTIFIER", "FULL_NAME", "ABBREVIATION"]
    DDI_TYPES = ["drug", "group", "brand", "drug_n"]

    def __init__(self, basepath, baseport = 9191, **kwargs):
        self.models = {}
        self.basepath = basepath
        self.corpus = kwargs.get("corpus")
        submodels = []
        self.baseport = baseport
        self.types = []
        if basepath.split("/")[-1].startswith("chemdner+ddi"):
            self.types = self.DDI_TYPES + self.CHEMDNER_TYPES + ["chemdner", "ddi"]
        elif basepath.split("/")[-1].startswith("ddi"):
            self.types = self.DDI_TYPES + ["all"]
        elif basepath.split("/")[-1].startswith("chemdner") or basepath.split("/")[-1].startswith("cemp"):
            self.types = ["all"] + self.CHEMDNER_TYPES
        elif basepath.split("/")[-1].startswith("gpro"):
            self.types = self.GPRO_TYPES + ["all"]
        self.basemodel = StanfordNERModel(self.basepath, "all")

    def train_types(self):
        """
        Train models for each subtype of entity, and a general model.
        :param types: subtypes of entities to train individual models, as well as a general model
        """
        self.basemodel.load_data(self.corpus, list(feature_extractors.keys()), subtype="all")
        for t in self.types:
            typepath = self.basepath + "_" + t
            model = StanfordNERModel(typepath, subtypes=self.basemodel.subtypes)
            model.copy_data(self.basemodel, t)
            logging.info("training subtype %s" % t)
            model.train()
            self.models[t] = model

    def load_models(self):
        for i, t in enumerate(self.types):
            model = StanfordNERModel(self.basepath + "_" + t, t, subtypes=self.basemodel.subtypes)
            model.load_tagger(self.baseport + i)
            self.models[t] = model

    def process_type(self, modelst, t, corpus, basemodel, basepath, port):
        # load data only for one model since this takes at least 5 minutes each time
        logging.debug("{}: copying data...".format(t))
        modelst.copy_data(basemodel)
        #logging.debug("pre test %s" % model)
        logging.debug("{}: testing...".format(t))
        res = modelst.test(corpus, port)
        logging.info("{}:done...".format(t))
        return res

    def test_types(self, corpus):
        """
        Classify the corpus with multiple classifiers from different subtypes
        :return ResultSetNER object with the results obtained for the models
        """
        # TODO: parallelize
        results = ResultSetNER(corpus, self.basepath)
        self.basemodel.load_data(corpus, list(feature_extractors.keys()))
        all_results = []
        tasks = [(self.models[t], t, corpus, self.basemodel, self.basepath, self.baseport + i) for i, t in enumerate(self.types)]

        all_results = []
        for t in tasks:
            r = self.process_type(*t)
            all_results.append(r)
        logging.info("adding results...")
        for res, i in enumerate(all_results):
            #logging.debug("adding these results: {}".format(self.types[i]))
            results.add_results(res)
        return results

