import codecs
import time
import logging
import os
import xml.etree.ElementTree as ET

from text.corpus import Corpus
from text.document import Document
from text.sentence import Sentence


class ChebiCorpus(Corpus):
    """Corpus based on the ChEBI annotated patents"""
    def __init__(self, corpusdir, **kwargs):
        super(ChebiCorpus, self).__init__(corpusdir, **kwargs)
        self.subtypes = ['CHEMICAL', 'FORMULA', 'LIGAND', 'ONT', 'CLASS']

    def get_docs(self, basedir):
        docs = []
        trainfiles = [(f, basedir + '/' + f + "/source.xml") for f in os.listdir(basedir)
                      if os.path.isdir(basedir + '/' + f)]
        for f in trainfiles:
            with open(f[1], 'r') as xml:
                root = ET.fromstring(xml.read())
                docid = f[0]
                body = root.find("BODY")
                docs.append((docid, body))
        return docs

    def get_paragraphs(self, doc, ):
        sents = []
        for div in doc[1].findall("DIV"):
            for p in div.findall("P"):
                if p.text.strip() != "" or len(p) != 0:
                    sid = "p " + str(len(sents))
                    sents.append((sid, p))
        return sents

    def load_corpus(self, corenlpserver):
        docs = self.get_docs(self.path)
        total = len(docs)
        current = 0
        time_per_abs = []
        ts = set()
        for f in docs:
            logging.debug('%s:%s/%s', f[0], current + 1, total)
            current += 1
            #parse DDI corpus file
            t = time.time()
            #print root.tag
            docid = f[0] # TODO: actually each paragraph should be it's own documents, that should help offset issues
            doctext = ""
            doc_sentences = [] # get the sentences of this document
            doc_offset = 0 # offset of the current sentence relative to the document
            sents = self.get_paragraphs(f)
            for p in sents:
                logging.debug("processing {}".format(p[0]))
                senttext = p[1].text.replace("\n", " ")
                for ne in p[1].findall("ne"):
                    #doctext += ne.text
                    senttext += ne.text
                    if ne.tail:
                        #doctext += ne.tail
                        senttext += ne.tail.replace("\n", " ")
                #logging.debug(senttext)
                #this_sentence = Sentence(senttext, offset=doc_offset, sid=p[0], did=docid)
                doctext += senttext + "\n"
                doc_offset = len(doctext)
                #doc_sentences.append(this_sentence)
                    #logging.info(len(doc_sentences))
            newdoc = Document(doctext, process=False, did=docid, ssplit=True)
            #newdoc.sentences = doc_sentences[:]
            newdoc.process_document(corenlpserver, "biomedical")
            #logging.info(len(newdoc.sentences))
            self.documents[newdoc.did] = newdoc
            #for s in self.documents[newdoc.did].sentences:
            #    logging.debug("sentence {} has {} tokens".format(s.sid, len(s.tokens)))
            #    logging.debug([(t.start, t.end) for t in s.tokens])
            abs_time = time.time() - t
            time_per_abs.append(abs_time)
            logging.info("%s sentences, %ss processing time" % (len(newdoc.sentences), abs_time))
        abs_avg = sum(time_per_abs)*1.0/len(time_per_abs)
        logging.info("average time per abstract: %ss" % abs_avg)

    def load_annotations(self, ann_dir, entitytype="chemical"):
        docs = self.get_docs(ann_dir)
        total = len(docs)
        current = 0
        time_per_abs = []
        ts = set()
        for f in docs:
            logging.debug('%s:%s/%s', f[0], current + 1, total)
            current += 1
            #parse DDI corpus file
            t = time.time()
            #print root.tag
            docid = f[0]
            doctext = ""
            doc_sentences = [] # get the sentences of this document
            doc_offset = 0 # offset of the current sentence relative to the document
            sents = self.get_paragraphs(f)
            for p in sents:
                #this_sentence = self.documents[docid].get_sentence(p[0])

                # logging.debug("sentence {} has {} tokens".format(this_sentence.sid, len(this_sentence.tokens)))
                # logging.debug([(t.start, t.end) for t in this_sentence.tokens])
                # logging.debug(this_sentence.sid)
                doctext += p[1].text.replace("\n", " ")
                nentity = 0
                for ne in p[1].findall("ne"):

                    # logging.debug("found entity: {} {}-{}".format(ne.text, len(senttext),
                    #                                              len(senttext) + len(ne.text)))
                    #print "text before:", senttext
                    destart = len(doctext)
                    deend = destart + len(ne.text)
                    this_sentence = self.documents[docid].find_sentence_containing(destart, deend,
                                                                               chemdner=False)
                    if this_sentence is None:
                        this_sentence = self.documents[docid].find_sentence_containing(destart+1, deend,
                                                                               chemdner=False)
                        if this_sentence is None:
                            # print "sentence not found!", destart, deend, ne.text
                            # print "doc has {} sentences".format(len(self.documents[docid].sentences))
                            continue
                    realoffset = this_sentence.offset
                    if this_sentence.offset-1 < len(doctext) and doctext[this_sentence.offset-1] == self.documents[docid].text[this_sentence.offset]:
                            realoffset =  this_sentence.offset - 1


                    estart = len(doctext) - realoffset
                    eend = estart + len(ne.text)

                    eid = this_sentence.tag_entity(estart, eend,
                                             ne.get("type"), text=ne.text)
                    if eid is None:
                        # print doctext[this_sentence.offset:]
                        #print self.documents[docid].text[this_sentence.offset:this_sentence.offset + len(this_sentence.text)]
                        logging.debug("got none: {} {}".format(this_sentence.sid, ne.text))
                        continue
                    nentity += 1
                    doctext += ne.text
                    if ne.tail:
                        doctext += ne.tail.replace("\n", " ")
                doc_sentences.append(p[0])

def main():
    c = ChebiCorpus("ChebiPatents/")
    c.load_corpus("")

if __name__ == "__main__":
    main()

