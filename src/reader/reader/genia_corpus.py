__author__ = 'Andre'
import codecs
import time
import logging
import sys
import os
from bs4 import BeautifulSoup
import progressbar as pb
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '../..'))
from text.corpus import Corpus
from text.document import Document
from text.sentence import Sentence

type_match = {"G#protein": "protein",
              "G#DNA": "dna"}

class GeniaCorpus(Corpus):
    def __init__(self, corpusdir, **kwargs):
        super(GeniaCorpus, self).__init__(corpusdir, **kwargs)
        self.subtypes = ["protein", "DNA"]

    def load_corpus(self, corenlpserver, process=True):

        soup = BeautifulSoup(codecs.open(self.path, 'r', "utf-8"), 'html.parser')
        docs = soup.find_all("article")
        widgets = [pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA(), ' ', pb.Timer()]
        pbar = pb.ProgressBar(widgets=widgets, maxval=len(docs)).start()
        n_lines = 1
        time_per_abs = []
        for doc in docs:
            did = "GENIA" + doc.articleinfo.bibliomisc.text.split(":")[1]
            title = doc.title.sentence.get_text()
            sentences = doc.abstract.find_all("sentence")
            doc_sentences = []
            doc_text = title + " "
            doc_offset = 0
            for si, s in enumerate(sentences):
                t = time.time()
                stext = s.get_text()
                sid = did + ".s" + str(si)
                doc_text += stext + " "
                this_sentence = Sentence(stext, offset=doc_offset, sid=sid, did=did)
                doc_offset = len(doc_text)
                doc_sentences.append(this_sentence)
            newdoc = Document(doc_text, process=False, did=did)
            newdoc.sentences = doc_sentences[:]
            newdoc.process_document(corenlpserver, "biomedical")
            #logging.info(len(newdoc.sentences))
            self.documents[newdoc.did] = newdoc
            abs_time = time.time() - t
            time_per_abs.append(abs_time)
            logging.debug("%s sentences, %ss processing time" % (len(newdoc.sentences), abs_time))
            pbar.update(n_lines)
            n_lines += 1
        pbar.finish()
        abs_avg = sum(time_per_abs)*1.0/len(time_per_abs)
        logging.info("average time per abstract: %ss" % abs_avg)


    def load_annotations(self, ann_dir, etype, ptype):
        time_per_abs = []
        skipped = 0
        notskipped = 0
        soup = BeautifulSoup(codecs.open(self.path, 'r', "utf-8"), 'html.parser')
        docs = soup.find_all("article")
        all_entities = {}
        for doc in docs:
            did = "GENIA" + doc.articleinfo.bibliomisc.text.split(":")[1]
            title = doc.title.find_all("sentence")
            # TODO: title also has annotations...
            sentences = doc.abstract.find_all("sentence")
            for si, s in enumerate(sentences):
                stext = s.get_text()
                sid = did + ".s" + str(si)
                this_sentence = self.documents[did].get_sentence(sid)
                sentities = s.find_all("cons", recursive=False)
                lastindex = 0
                for ei, e in enumerate(sentities):
                    estart = stext.find(e.text, lastindex)
                    eend = estart + len(e.text)
                    etext = stext[estart:eend]
                    # sems = e.get("sem")
                    sem = e.get("sem")
                    if sem.startswith("("):
                        #TODO: Deal with overlapping entities
                        continue
                    entity_type = sem.split("_")[0]
                    if etype == "all" or type_match.get(entity_type, "entity") == etype:
                        eid = this_sentence.tag_entity(estart, eend, type_match.get(entity_type, "entity"),
                                                     text=e.text)
                        if eid is None:
                            print "did not add this entity: {}".format(e.text)
                        # print e.text
                        notskipped += 1

                    t = sem.split("_")[0]
                    if t not in all_entities:
                        all_entities[t] = []
                    all_entities[t].append(etext)
                    #if sem is not None and sem.startswith("G#protein"):
                    #    print e.text, "|", etext, eindex, stext[0:20]
                    lastindex = estart
        #for s in all_entities:
        #    print s, len(all_entities[s])


def get_genia_gold_ann_set(goldann, etype):
    gold_offsets = set()
    soup = BeautifulSoup(codecs.open(goldann, 'r', "utf-8"), 'html.parser')
    docs = soup.find_all("article")
    all_entities = {}
    for doc in docs:
        did = "GENIA" + doc.articleinfo.bibliomisc.text.split(":")[1]
        title = doc.title.sentence.get_text()
        doc_text = title + " "
        doc_offset = 0
        sentences = doc.abstract.find_all("sentence")
        for si, s in enumerate(sentences):
            stext = s.get_text()
            sentities = s.find_all("cons", recursive=False)
            lastindex = 0
            for ei, e in enumerate(sentities):
                estart = stext.find(e.text, lastindex) + doc_offset # relative to document
                eend = estart + len(e.text)
                sem = e.get("sem")
                if sem.startswith("("):
                    #TODO: Deal with overlapping entities
                    continue
                entity_type = sem.split("_")[0]
                if etype == "all" or type_match.get(entity_type) == etype:
                    gold_offsets.add((did, estart, eend, e.text))
                # etext = doc_text[estart:eend]
                # logging.info("gold annotation: {}".format(e.text))

            doc_text += stext + " "
            doc_offset = len(doc_text)
    return gold_offsets, None

def main():
    logging.basicConfig(level=logging.DEBUG)
    c = GeniaCorpus(sys.argv[1])
    c.load_corpus("")
    c.load_annotations(sys.argv[1])
if __name__ == "__main__":
    main()