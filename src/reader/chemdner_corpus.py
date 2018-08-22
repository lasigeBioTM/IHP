import codecs
import time
import sys
import logging
import argparse
import pickle
from operator import itemgetter
from pycorenlp import StanfordCoreNLP
import progressbar as pb
from subprocess import check_output

from text.corpus import Corpus
from text.document import Document
from config import config

class ChemdnerCorpus(Corpus):
    """Chemdner corpus from BioCreative IV and V"""
    def __init__(self, corpusdir, **kwargs):
        super(ChemdnerCorpus, self).__init__(corpusdir, **kwargs)
        self.subtypes = ["IDENTIFIER", "MULTIPLE", "FAMILY", "FORMULA", "SYSTEMATIC", "ABBREVIATION", "TRIVIAL"]

    def load_corpus(self, corenlpserver, process=True):
        """Load the CHEMDNER corpus file on the dir element"""
        # open filename and parse lines
        total_lines = sum(1 for line in open(self.path))
        widgets = [pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA(), ' ', pb.Timer()]
        pbar = pb.ProgressBar(widgets=widgets, maxval=total_lines).start()
        n_lines = 1
        time_per_abs = []
        with codecs.open(self.path, 'r', "utf-8") as inputfile:
            for line in inputfile:
                t = time.time()
                # each line is PMID  title   abs
                tsv = line.split('\t')
                doctext = tsv[2].strip().replace("<", "(").replace(">", ")")
                newdoc = Document(doctext, process=False,
                                  did=tsv[0], title=tsv[1].strip())
                newdoc.sentence_tokenize("biomedical")
                if process:
                    newdoc.process_document(corenlpserver, "biomedical")
                self.documents[newdoc.did] = newdoc
                n_lines += 1
                abs_time = time.time() - t
                time_per_abs.append(abs_time)
                pbar.update(n_lines+1)
        pbar.finish()
        abs_avg = sum(time_per_abs)*1.0/len(time_per_abs)
        logging.info("average time per abstract: %ss" % abs_avg)

    def load_annotations(self, ann_dir, entitytype="chemical"):
        # total_lines = sum(1 for line in open(ann_dir))
        # n_lines = 1
        logging.info("loading annotations file...")
        with codecs.open(ann_dir, 'r', "utf-8") as inputfile:
            for line in inputfile:
                # logging.info("processing annotation %s/%s" % (n_lines, total_lines))
                pmid, doct, start, end, text, chemt = line.strip().split('\t')
                #pmid = "PMID" + pmid
                if pmid in self.documents:
                    if entitytype == "all" or entitytype == "chemical" or entitytype == chemt:
                        self.documents[pmid].tag_chemdner_entity(int(start), int(end),
                                                             chemt, text=text, doct=doct)
                else:
                    logging.info("%s not found!" % pmid)

def write_chemdner_files(results, models, goldset, ths, rules):
    """ results files for CHEMDNER CEMP and CPD tasks"""
    print("saving results to {}".format(results.path + ".tsv"))
    with codecs.open(results.path + ".tsv", 'w', 'utf-8') as outfile:
        cpdlines, max_entities = results.corpus.write_chemdner_results(models, outfile, ths, rules)
    cpdlines = sorted(cpdlines, key=itemgetter(2))
    with open(results.path + "_cpd.tsv", "w") as cpdfile:
        for i, l in enumerate(cpdlines):
            if l[2] == 0:
                cpdfile.write("{}_{}\t0\t{}\t1\n".format(l[0], l[1], i+1))
            else:
                cpdfile.write("{}_{}\t1\t{}\t{}\n".format(l[0], l[1], i+1, l[2]*1.0/max_entities))

def run_chemdner_evaluation(goldstd, results, format=""):
    """
    Use the official BioCreative evaluation script (should be installed in the system)
    :param goldstd: Gold standard file path
    :param results: Results file path
    :param: format option
    :return: Output of the evaluation script
    """
    cem_command = ["bc-evaluate", results, goldstd]
    if format != "":
        cem_command = cem_command[:1] + [format] + cem_command[1:]
    r = check_output(cem_command)
    return r


def get_chemdner_gold_ann_set(goldann="CHEMDNER/CHEMDNER_TEST_ANNOTATION/chemdner_ann_test_13-09-13.txt"):
    """
    Load the CHEMDNER annotations to a set
    :param goldann: Path to CHEMDNER annotation file
    :return: Set of gold standard annotations
    """
    with codecs.open(goldann, 'r', 'utf-8') as goldfile:
            gold = goldfile.readlines()
    goldlist = []
    for line in gold:
        #pmid, T/A, start, end
        x = line.strip().split('\t')
        goldlist.append((x[0], x[1] + ":" + x[2] + ":" + x[3], '1'))
    #print goldlist[0:2]
    goldset = set(goldlist)
    return goldset, None

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("actions", default="classify",  help="Actions to be performed.",
                      choices=["load_corpus"])
    parser.add_argument("--goldstd", default="", dest="goldstd", nargs="+",
                      help="Gold standard to be used. Will override corpus, annotations",
                      choices=list(config.paths.keys()))
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
    parser.add_argument("--doctype", dest="doctype", help="type of document to be considered", default="all")
    parser.add_argument("--annotated", action="store_true", default=False, dest="annotated",
                      help="True if the input has <entity> tags.")
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
    logging.info("Processing action {0} on {1}".format(options.actions, options.goldstd))

    # set configuration variables based on the goldstd option if the corpus has a gold standard,
    # or on corpus and annotation options
    # pre-processing options
    if options.actions == "load_corpus":
        if len(options.goldstd) > 1:
            print("load only one corpus each time")
            sys.exit()
        options.goldstd = options.goldstd[0]
        corpus_format = config.paths[options.goldstd]["format"]
        corpus_path = config.paths[options.goldstd]["text"]
        corpus_ann = config.paths[options.goldstd]["annotations"]
        corenlp_client = StanfordCoreNLP('http://localhost:9000')
        if corpus_format == "chemdner":
            corpus = ChemdnerCorpus(corpus_path)
            #corpus.save()
            if options.goldstd == "chemdner_traindev":
                # merge chemdner_train and chemdner_dev
                tpath = config.paths["chemdner_train"]["corpus"]
                tcorpus = pickle.load(open(tpath, 'rb'))
                dpath = config.paths["chemdner_dev"]["corpus"]
                dcorpus = pickle.load(open(dpath, 'rb'))
                corpus.documents.update(tcorpus.documents)
                corpus.documents.update(dcorpus.documents)
            elif options.goldstd == "cemp_test_divide":
                logging.info("loading corpus %s" % corpus_path)
                corpus.load_corpus(corenlp_client, process=False)
                docs = list(corpus.documents.keys())
                step = int(len(docs)/10)
                logging.info("step: {}".format(str(step)))
                for i in range(10):
                    logging.info("processing cemp_test{}: {} - {}".format(str(i), int(step*i), int(step*i+step)))
                    sub_corpus_path = config.paths["cemp_test" + str(i)]["corpus"]
                    sub_corpus = ChemdnerCorpus(sub_corpus_path)
                    sub_docs = docs[int(step*i):int(step*i+step)]
                    for di, d in enumerate(sub_docs):
                        logging.info("fold {}: processing {}/{}".format(i, di, step))
                        sub_corpus.documents[d] = corpus.documents[d]
                        del corpus.documents[d]
                        sub_corpus.documents[d].process_document(corenlp_client)
                    sub_corpus.save()

if __name__ == "__main__":
    main()