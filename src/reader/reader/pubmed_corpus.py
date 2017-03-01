import logging
import time
import progressbar as pb
from text.corpus import Corpus
from pubmed import PubmedDocument

class PubmedCorpus(Corpus):
    """
        Corpus composed by Pubmed documents
    """
    def __init__(self, corpusdir, pmidlist, **kwargs):
         super(PubmedCorpus, self).__init__(corpusdir, **kwargs)
         self.pmids = pmidlist

    def load_corpus(self, corenlpserver, process=True):
        """
        Use the PubMed web services to retrieve the title and abstract of each PMID
        :param corenlpserver:
        :param process:
        :return:
        """
        time_per_abs = []
        widgets = [pb.Percentage(), ' ', pb.Bar(), ' ', pb.AdaptiveETA(), ' ', pb.Timer()]
        pbar = pb.ProgressBar(widgets=widgets, maxval=len(self.pmids), redirect_stdout=True).start()
        for i, pmid in enumerate(self.pmids):
            t = time.time()
            newdoc = PubmedDocument(pmid)
            if newdoc.abstract == "":
                logging.info("ignored {} due to the fact that no abstract was found".format(pmid))
                continue
            newdoc.process_document(corenlpserver, "biomedical")
            self.documents["PMID" + pmid] = newdoc
            abs_time = time.time() - t
            time_per_abs.append(abs_time)
            pbar.update(i+1)
        pbar.finish()
        abs_avg = sum(time_per_abs)*1.0/len(time_per_abs)
        logging.info("average time per abstract: %ss" % abs_avg)