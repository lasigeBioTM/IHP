import codecs
import logging
import xml.etree.ElementTree as ET
import os
import sys
import progressbar as pb
import time
import progressbar as pb
from text.corpus import Corpus
from text.document import Document
from text.sentence import Sentence

type_match = {"DNA": "protein",
              "protein": "protein"}

class JNLPBACorpus(Corpus):
    def __init__(self, corpusdir, **kwargs):
        super(JNLPBACorpus, self).__init__(corpusdir, **kwargs)
        self.subtypes = ["protein", "DNA"]

    def load_corpus(self, corenlpserver, process=True):
        widgets = [pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA(), ' ', pb.Timer()]
        nlines = 0
        with open(self.path) as f:
            for nlines, l in enumerate(f):
                pass
        print(nlines)
        pbar = pb.ProgressBar(widgets=widgets, maxval=nlines).start()
        with codecs.open(self.path, 'r', "utf-8") as corpusfile:
            doc_text = ""
            sentences = []
            for i,l in enumerate(corpusfile):
                if l.startswith("###"): # new doc
                    if doc_text != "":
                        logging.debug("creating document...")
                        newdoc = Document(doc_text, process=False, did=did)
                        newdoc.sentences = sentences[:]
                        newdoc.process_document(corenlpserver, "biomedical")
                        # logging.info(len(newdoc.sentences))
                        self.documents[newdoc.did] = newdoc
                    did = "JNLPBA" + l.strip().split(":")[-1]
                    logging.debug("starting new document:" + did)
                    sentence_text = ""
                    doc_offset = 0
                    sentences = []
                elif l.strip() == "" and sentence_text != "": # new sentence
                    logging.debug("creating sentence...")
                    sid = did + ".s" + str(len(sentences))
                    this_sentence = Sentence(sentence_text, offset=doc_offset, sid=sid, did=did)
                    doc_offset += len(sentence_text) + 1
                    doc_text += sentence_text + " "
                    sentences.append(this_sentence)
                else:
                    logging.debug(str(i) + "/" + str(l))
                    t = l.strip().split("\t")
                    if sentence_text != " ":
                        sentence_text += " "
                    #if t[1] == "B-protein"
                    sentence_text += t[0]
                pbar.update(i)
            pbar.finish()

    def load_annotations(self, ann_dir, etype, ptype):
        with codecs.open(ann_dir, 'r', "utf-8") as annfile:
            sentences = []
            for l in annfile:
                if l.startswith("###"):  # new doc
                    did = "JNLPBA" + l.strip().split(":")[-1]
                    sentence_text = ""
                elif l.strip() == "" and sentence_text != "":  # new sentence
                    sid = did + ".s" + str(len(sentences))
                    this_sentence = self.documents[did].get_sentence(sid)
                    sentences.append(this_sentence)
                else:
                    t = l.strip().split("\t")
                    if sentence_text != " ":
                        sentence_text += " "
                    if t[1] == "B-" + etype:
                        estart = len(sentence_text)
                        eend = estart + len(t[0])
                        entity_text = t[0]
                        added = False
                    elif t[1] == "I-" + etype:
                        eend += 1 + len(t[0])
                        entity_text += " " + t[0]
                    else: # not B- I-
                        if not added:
                            eid = this_sentence.tag_entity(estart, eend, etype,
                                                           text=entity_text)
                            if eid is None:
                                print("did not add this entity: {}".format(entity_text))
                            added = True
                    if sentence_text != "":
                        sentence_text += " "
                    sentence_text += t[0]

def get_genia_gold_ann_set(goldann, etype):
    gold_offsets = set()
    with codecs.open(goldann, 'r', "utf-8") as annfile:
        sentences = []
        for l in annfile:
            if l.startswith("###"):  # new doc
                did = "JNLPBA" + l.strip().split(":")[-1]
                sentence_text = ""
            elif l.strip() == "" and sentence_text != "":  # new sentence
                sid = did + ".s" + str(len(sentences))
            else:
                t = l.strip().split("\t")
                if sentence_text != " ":
                    sentence_text += " "
                if t[1] == "B-" + etype:
                    estart = len(sentence_text)
                    eend = estart + len(t[0])
                    entity_text = t[0]
                    added = False
                elif t[1] == "I-" + etype:
                    eend += 1 + len(t[0])
                    entity_text += " " + t[0]
                else:  # not B- I-
                    if not added:
                        gold_offsets.add((did, estart, eend, entity_text))
                        added = True
                if sentence_text != "":
                    sentence_text += " "
                sentence_text += t[0]