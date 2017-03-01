import codecs
import time
import logging
import xml.etree.ElementTree as ET
import os
import sys

from text.corpus import Corpus
from text.document import Document
from text.sentence import Sentence


class DDICorpus(Corpus):
    """
    DDI corpus used for NER and RE on the SemEval DDI tasks of 2011 and 2013.
    self.path is the base directory of the files of this corpus.
    Each file is a document, DDI XML format, sentences already separated.
    """
    def __init__(self, corpusdir, **kwargs):
        super(DDICorpus, self).__init__(corpusdir, **kwargs)
        self.subtypes = ["drug", "group", "brand", "drug_n"]

    def load_corpus(self, corenlpserver):
        # self.path is the base directory of the files of this corpus
        trainfiles = [self.path + '/' + f for f in os.listdir(self.path) if f.endswith('.xml')]
        total = len(trainfiles)
        current = 0
        time_per_abs = []
        for f in trainfiles:
            logging.debug('%s:%s/%s', f, current + 1, total)
            current += 1
            with open(f, 'r') as xml:
                #parse DDI corpus file
                t = time.time()
                root = ET.fromstring(xml.read())
                doctext = ""
                did = root.get('id')
                doc_sentences = [] # get the sentences of this document
                doc_offset = 0 # offset of the current sentence relative to the document
                for sentence in root.findall('sentence'):
                    sid = sentence.get('id')
                    #logging.info(sid)
                    text = sentence.get('text')
                    text = text.replace('\r\n', '  ')
                    doctext += " " + text # generate the full text of this document
                    this_sentence = Sentence(text, offset=doc_offset, sid=sid, did=did)
                    doc_offset = len(doctext)
                    doc_sentences.append(this_sentence)
                #logging.info(len(doc_sentences))
                newdoc = Document(doctext, process=False, did=did)
                newdoc.sentences = doc_sentences[:]
                newdoc.process_document(corenlpserver, "biomedical")
                #logging.info(len(newdoc.sentences))
                self.documents[newdoc.did] = newdoc
                abs_time = time.time() - t
                time_per_abs.append(abs_time)
                logging.info("%s sentences, %ss processing time" % (len(newdoc.sentences), abs_time))
        abs_avg = sum(time_per_abs)*1.0/len(time_per_abs)
        logging.info("average time per abstract: %ss" % abs_avg)

    def getOffsets(self, offset):
        # check if its just one offset per entity or not
        # add 1 to offset end to agree with python's indexes
        offsets = []
        offsetList = offset.split(';')
        for o in offsetList:
            offsets.append(int(o.split('-')[0]))
            offsets.append(int(o.split('-')[1])+1)

        #if len(offsets) > 2:
        #    print "too many offsets!"
            #sys.exit()
        return offsets

    def load_annotations(self, ann_dir, etype):
        trainfiles = [ann_dir + '/' + f for f in os.listdir(ann_dir) if f.endswith('.xml')]
        total = len(trainfiles)
        current = 0
        time_per_abs = []
        logging.info("loading annotations...")
        for f in trainfiles:
            logging.debug('%s:%s/%s', f, current + 1, total)
            current += 1
            with open(f, 'r') as xml:
                #parse DDI corpus file
                t = time.time()
                root = ET.fromstring(xml.read())
                did = root.get('id')
                for sentence in root.findall('sentence'):
                    sid = sentence.get('id')
                    this_sentence = self.documents[did].get_sentence(sid)
                    if this_sentence is None:
                        print did, sid, "sentence not found!"
                        for entity in sentence.findall('entity'):
                            print entity.get('charOffset'), entity.get("type")
                        print [s.sid for s in self.documents[did].sentences]
                        sys.exit()
                        #continue
                    for entity in sentence.findall('entity'):
                        eid = entity.get('id')
                        entity_offset = entity.get('charOffset')
                        offsets = self.getOffsets(entity_offset)
                        entity_type = entity.get("type")
                        if etype == "chemical" or etype == "all" or (etype != "all" and etype == entity_type):
                            this_sentence.tag_entity(offsets[0], offsets[-1], entity_type,
                                                 text=entity.get("text"))