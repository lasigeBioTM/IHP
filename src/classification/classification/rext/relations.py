#!/usr/bin/env python -W ignore::ModuleDeprecationWarning
#train and evaluate ML model for DDI classification based on the DDI corpus
import logging
import os
import pickle
import sys
import tarfile
import time
from optparse import OptionParser

import kernelmodels

TRUE_DDI = 'trueDDI'
#shallow linguist kernel prediction using jsre
SLK_PRED = 'slk_pred'
#subset tree kernel using SVM-Light-TK
SST_PRED = 'sst_pred'
#ddi_ensemble trained with the result of the other classifiers
ENSEMBLE_PRED = "ensemble_pred"

FINAL_PRED = 'final_pred'
allclassifiers = [SLK_PRED, SST_PRED, "all", ""]


def getSentenceID(pair):
    return '.'.join(pair.split('.')[:-1])


def get_docs_pairs(allpairs, docs):
    return dict([(p, allpairs[p])
                 for p in allpairs
                 if getSentenceID(p) in docs])


def get_pairs_dic(kernelpairs, docs):
    return dict([(p, kernelpairs[p])
                 for p in kernelpairs
                 if getSentenceID(p) in docs])




def train(docs, kernelpairs, classifiers, dditype="all", tag="", backup=False):
    tempfiles = []
    excludesentences = []
    #data = ddi_sentences.build_data_frame(docs)


    #logging.info("excluding %s/%s sentences from the train data",
    #             len(excludesentences), len(docs))

    if SLK_PRED in classifiers:
        logging.info("**Training SLK classifier %s ..." % (tag,))
        #trainpairdic = ddi_kernels.fromddiDic(traindocs)

        kernelmodels.generatejSREdata(kernelpairs, docs, tag + "ddi_train_jsre.txt",
                                      dditype=dditype, excludesentences=excludesentences,
                                      train=True)
        kernelmodels.trainjSRE(tag + "ddi_train_jsre.txt", tag + "ddi_train_slk.model")
        logging.info("done.")
        #logging.info("pred: %s \ntest_y: %s", labels, test_y)
        #tempfiles.append(ddi_kernels.basedir + tag + "ddi_train_jsre.txt")
        #tempfiles.append(ddi_kernels.basedir + tag + "ddi_train_slk.model")
    if SST_PRED in classifiers:
        logging.info("****Training SST classifier %s ..." % (tag,))
        kernelmodels.trainSVMTK(docs, kernelpairs, model=tag + "ddi_train_sst.model",
                                dditype=dditype, excludesentences=excludesentences)
        tempfiles.append("ddi_models/" + tag + "ddi_train_sst.model")
        logging.info("done.")
    print tag + " training complete"
    if backup:
        print "backing up these files:", tempfiles
        backupFiles("train_results_", tempfiles)
    return tempfiles


def test(docs, allpairs, kernelpairs, classifiers=[SLK_PRED, SST_PRED],
         dditype=type, tag="", backup=False, printstd=False):
    #expects: ddi_sentences.md
    #data =  ddi_train_slk.model, ddi_train_sst.model
    tempfiles = []
    excludesentences = []

    if SLK_PRED in classifiers:
        logging.info("**Testing SLK classifier %s ..." % (tag,))
        #testpairdic = ddi_kernels.fromddiDic(testdocs)
        kernelmodels.generatejSREdata(kernelpairs, docs, tag + "ddi_test_jsre.txt", dditype=dditype,
                                      excludesentences=excludesentences)
        kernelmodels.testjSRE(tag + "ddi_test_jsre.txt", tag + "ddi_test_result.txt",
                              model=tag + "ddi_train_slk.model")
        allpairs = kernelmodels.getjSREPredicitons(tag + "ddi_test_jsre.txt", tag + "ddi_test_result.txt",
                                                   allpairs, kernelpairs, dditype=dditype)
        tempfiles.append(kernelmodels.basedir + tag + "ddi_test_jsre.txt")
        tempfiles.append(kernelmodels.basedir + tag + "ddi_test_result.txt")

    if SST_PRED in classifiers:
        logging.info("****Testing SST classifier %s ..." % (tag,))
        allpairs = kernelmodels.testSVMTK(docs, kernelpairs, allpairs, dditype=dditype,
                                          model=tag + "ddi_train_sst.model", tag=tag,
                                          excludesentences=excludesentences)
    return tempfiles, allpairs


def backupFiles(name, fileslist):
    tar = tarfile.open(name + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".tar.gz", "w:gz")
    for f in fileslist:
        tar.add(f)
        os.remove(f)
    tar.close()

def main():
    parser = OptionParser(usage='''train and evaluate ML model
                                   for DDI classification based on the DDI corpus''')
    parser.add_option("-f", "--file", dest="file",  action="store", default="ddi",
                      help="Pickle file to load/store the data")
    parser.add_option("-d", "--dir", action="store", dest="dir", type="string", default=None,
                      help="Corpus directory with XML files. Depends on the action option")
    parser.add_option("--reload", action="store", default="", dest="reload",
                      help="Reload corpus - sentences and/or treekernel")
    parser.add_option("--log", action="store", dest="loglevel", type="string", default="WARNING",
                      help="Log level")
    parser.add_option("--nfolds", action="store", dest="nfolds", type="int", default=5,
                      help="Number of cross-validation folds")
    parser.add_option("--ssm", action="store", dest="ssm", type="float", default=0.0,
                      help="SSM threshold")
    parser.add_option("--ssmtype", action="store", dest="ssmtype", type="string", default="simgic",
                      help="SSM type to use")
    parser.add_option("-a", "--action", action="store", dest="action",
                      type="string", default="loadcorpus",
                      help="cv, train, test, classify, optimize, loadcorpus, results")
    parser.add_option("--useclassifiers", action="store", dest="use", type="string",
                      default="sentence_pred slk_pred sst_pred", help="classifiers")
    parser.add_option("--vote", action="store", dest="vote", type="float", default=1,
                      help="minimum number of votes to be considered DDI")
    parser.add_option("--lowmax", action="store", dest="lowmax", type="float", default=0,
                      help="maximum number of pairs for the low group")
    parser.add_option("--nobackup", action="store_false", default=True, dest="backup",
                      help="do not backup files")
    parser.add_option("--extra", action="store", dest="extra", type="string", default="",
                      help="extra string to be printed for the log")
    parser.add_option("--types", action="store", dest="types", type="string",
                      default="all effect mechanism advise int",
                      help="ddi types to train - all, effect, mechanism, int, advise")
    parser.add_option("-e", "--ensemble", action="store", dest="ensemble",
                      type="string", default="mv",
                      help="mv: majority voting, ml_cv, ml_train, ml_test: machine learning")
    parser.add_option("-r", "--results", dest="res",  action="store", default="",
                      help="results file to load")
    (options, args) = parser.parse_args()

    numeric_level = getattr(logging, options.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % options.loglevel)

    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    logging.basicConfig(level=numeric_level, format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger().setLevel(numeric_level)
    logging.debug("debug test")
    logging.info("info test")
    logging.warning("warning test")

    for c in options.use.split(' '):
        if c not in allclassifiers:
            print "unknown classifier:", c
            sys.exit()

    #print "using:", options.use.split(' ')
    docs, allpairs, kernelpairs = load_data(options.reload.split(' '),
                                            options.file, options.action,
                                            options.dir)
    logging.info("finished loading data")
    dditypes = options.types.split(" ")

    print "=============================================================="
    print "experimental conditions:"
    print "action=" + options.action, "classifiers=" + options.use,
    print "nfolds=" + str(options.nfolds), "vote=" + str(options.vote),
    print "lowmax=" + str(options.lowmax), "backup=" + str(options.backup)
    print "ddi types=" + options.types, "results=", options.res
    print options.extra
    sys.stdout.flush()

    if options.res:
        usedclassifiers = set()
        print "loading results " + options.res
        prevres = pickle.load(open(options.res, 'rb'))
        for p in allpairs:
            if p in prevres:
                for cl in prevres[p]:
                    #if cl not in options.use.split(' '):
                    if cl != FINAL_PRED and cl != TRUE_DDI:
                        usedclassifiers.add(cl)
                        allpairs[p][cl] = prevres[p][cl]
            else:
                print "could not find %s in the results loaded" % p
        print "used these classifiers from results pickle:" + ' '.join(usedclassifiers)

    if options.action == "loadcorpus":
        sys.exit()

    elif options.action == 'train':
        #train classifiers
        #ddi_sentences.trainSentences(docs, outname = options.classifier)
        #train(docs, kernelpairs, classifiers=options.use.split(' '))
        tempfiles = []
        for t in dditypes:
            #train(docs, kernelpairs, classifiers, dditype="all", tag="", backup=False):
            tempfiles += train(docs, kernelpairs, options.use.split(' '),
                               tag=options.file + t + "_", dditype=t)
        #backupFiles("cv_results_", tempfiles)
        sys.exit()
    elif options.action == 'test':
        tempfiles = []
        for t in dditypes:
            testres = test(docs, allpairs, kernelpairs, classifiers=options.use.split(' '),
                           tag=options.file + t + "_", backup=False,
                           printstd=False, dditype=t)
            tempfiles += testres[0]
            allpairs = testres[1]
        #backupFiles("cv_results_", tempfiles)
        pickle.dump(allpairs, open("ddi_results/" + options.file + "_results" +
                               time.strftime("%Y%m%d_%H%M%S", time.localtime()) +
                               ".pickle", 'wb'))

    elif options.action == 'classify':
        sys.exit()
        #classify new data
        #classifySentences(docs, pipeline, classname = options.classifier)
    elif options.action == 'optimizeSentences':
        #optimize(docs)
        sys.exit()

    if options.action == "results" and options.use != "all":
        features = options.use.split(" ")
    else:
        features = "all"
    print features
    ddi_types.getTypePredictions(allpairs, docs, kernelpairs,
                                 method=options.ensemble, vote=options.vote,
                                 classifiers=features, source=options.dir)


if __name__ == "__main__":
    main()
