import codecs
import logging
import pickle

from chemdner_corpus import ChemdnerCorpus


class GproCorpus(ChemdnerCorpus):
    """Chemdner GPRO corpus from BioCreative V"""
    def __init__(self, corpusdir, **kwargs):
        super(GproCorpus, self).__init__(corpusdir, **kwargs)
        self.subtypes = ["NESTED", "IDENTIFIER", "FULL_NAME", "ABBREVIATION"]

    def load_corpus(self, corenlpserver):
        """
        Assume the corpus is already loaded as a ChemdnerCorpus
        Load the pickle and get the docs
        :param corenlpserver:
        :return:
        """
        ps = self.path.split("/")
        cemp_path = "data/chemdner_" + "_".join(ps[-1].split("_")[1:]) + ".pickle"
        corpus = pickle.load(open(cemp_path, 'rb'))
        self.documents = corpus.documents

    def load_annotations(self, ann_dir, etype="protein"):
        logging.info("loading annotations file {}...".format(ann_dir))
        with codecs.open(ann_dir, 'r', "utf-8") as inputfile:
            for line in inputfile:
                # logging.info("processing annotation %s/%s" % (n_lines, total_lines))
                pmid, doct, start, end, text, t, dbid = line.strip().split('\t')
                if dbid != "GPRO_TYPE_2" and pmid in self.documents:
                #if pmid in self.documents:
                #pmid = "PMID" + pmid
                    # For now, ignore the database ID information
                    #logging.debug("using this annotation: {}".format(text.encode("utf8")))
                    self.documents[pmid].tag_chemdner_entity(int(start), int(end),
                                                             t, text=text, doct=doct)
                elif pmid not in self.documents:
                    logging.info("%s not found!" % pmid)
