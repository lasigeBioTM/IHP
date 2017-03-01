import logging

import progressbar as pb
from reader.pubmed_corpus import PubmedCorpus
from mirna_base import MirbaseDB
from config import config


class TransmirCorpus(PubmedCorpus):
    """
        Corpus generated from the TransmiR database, using distant supervision
    """
    def __init__(self, corpusdir, **kwargs):
        self.mirbase = MirbaseDB(config.mirbase_path)
        self.mirbase.load_graph()
        self.mirnas = set()
        self.tfs = set()
        self.pairs = set()
        self.pmids = set()
        self.normalized_mirnas = set() # normalized to miRBase
        self.normalized_tfs = set() #normalized to maybe UniProt
        self.normalized_pairs = set()
        self.db_path = corpusdir
        self.load_database()
        super(TransmirCorpus, self).__init__(corpusdir, self.pmids, **kwargs)
        # TODO: use negatome

    def load_database(self):
        logging.info("Loading TransmiR database...")
        with open(self.db_path, 'r') as dbfile:
            for line in dbfile:
                tsv = line.strip().split("\t")
                if tsv[-1].lower() == "human":
                    tfname = tsv[0]
                    mirname = tsv[3]
                    func = tsv[5].split(";")
                    disease = tsv[6].split(";")
                    active = tsv[7]
                    pmid = tsv[8].split(";")
                    self.tfs.add(tfname) # uniform TF names
                    if not mirname.startswith("hsa-"):
                        mirname = "hsa-" + mirname
                    self.mirnas.add(mirname)
                    # for f in func:
                    #     funcs.add(f.strip())
                    # for d in disease:
                    #     if d != "see HMDD (http://cmbi.bjmu.edu.cn/hmdd)":
                    #         diseases.add(d.strip())
                    for p in pmid:
                        logging.info(p)
                        self.pmids.add(p.strip())
                    #mirnas[mirname] = (func, [d for d in disease if d != "see HMDD (http://cmbi.bjmu.edu.cn/hmdd)"])
                    #entries[(tfname, mirname)] = (active, pmid)

    def normalize_entities(self):
        logging.info("Normalizing entities...")
        for mir in self.mirnas:
            match = self.mirbase.map_label(mir)
            #if match > 0.6:
            self.normalized_mirnas.add(match[0])

    def load_annotations(self, db_path, etype):
        self.normalize_entities()

