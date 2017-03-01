from __future__ import division, absolute_import
#from nltk.stem.porter import PorterStemmer
#import jsonrpclib
#from simplejson import loads
import logging
import os
from subprocess import Popen, PIPE
import codecs
import xml.etree.ElementTree as ET
import sys
from config.config import geniass_path
from text.sentence import Sentence
from text.token2 import Token2
from text.pair import Pair, Pairs

from text.tlink import TLink
from other.dictionary import Dictionary, stopwords, removewords, go_words

gazette = Dictionary()

whitespace = [u"\u2002", u"\u2003", u"\u00A0", u"\u2009", u"\u200C", u"\u200D",
              u'\u2005', u'\u2009', u'\u200A']
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#porter = PorterStemmer()


def clean_whitespace(text):
    'replace all whitespace for a regular space " "'
    replacedtext = text
    for code in whitespace:
        replacedtext = replacedtext.replace(code, " ")
    return replacedtext


class Document(object):
    """A document is constituted by one or more sentences. It should have an ID and
    title. s0, the first sentence, is always the title sentence."""

    def __init__(self, text, process=False, doctype="biomedical", ssplit=False, **kwargs):
        self.text = text
        self.title = kwargs.get("title")
        self.sentences = kwargs.get("sentences", [])
        self.did = kwargs.get("did", "d0")
        self.invalid_sids = []
        self.title_sids = []
        self.pairs = Pairs()
        if ssplit:
            self.sentence_tokenize(doctype)
        if process:
            self.process_document(doctype)

    def sentence_tokenize(self, doctype):
        """
        Split the document text into sentences, add to self.sentences list
        :param doctype: Can be used in the future to choose different methods
        """
        # first sentence should be the title if it exists
        if self.title:
            sid = self.did + ".s0"
            self.sentences.append(Sentence(self.title, sid=sid, did=self.did))
        # inputtext = clean_whitespace(self.text)
        inputtext = self.text
        with codecs.open("/tmp/geniainput.txt", 'w', 'utf-8') as geniainput:
            geniainput.write(inputtext)
        current_dir = os.getcwd()
        os.chdir(geniass_path)
        geniaargs = ["./geniass", "/tmp/geniainput.txt", "/tmp/geniaoutput.txt"]
        Popen(geniaargs, stdout=PIPE, stderr=PIPE).communicate()
        os.chdir(current_dir)
        offset = 0
        with codecs.open("/tmp/geniaoutput.txt", 'r', "utf-8") as geniaoutput:
            for l in geniaoutput:
                stext = l.strip()
                if stext == "":
                    offset = self.get_space_between_sentences(offset)
                    continue
                sid = self.did + ".s" + str(len(self.sentences))
                self.sentences.append(Sentence(stext, offset=offset, sid=sid, did=self.did))
                offset += len(stext)
                offset = self.get_space_between_sentences(offset)

    def process_document(self, corenlpserver, doctype="biomedical"):
        """
        Process each sentence in the text (sentence split if there are no sentences) using Stanford CoreNLP
        :param corenlpserver:
        :param doctype:
        :return:
        """
        if len(self.sentences) == 0:
            # use specific sentence splitter
            self.sentence_tokenize(doctype)
        for s in self.sentences:
            #corenlpres = corenlpserver.raw_parse(s.text)
            corenlpres = corenlpserver.annotate(s.text.encode("utf8"), properties={
                'ssplit.eolonly': True,
                # 'annotators': 'tokenize,ssplit,pos,depparse,parse',
                'annotators': 'tokenize,ssplit,pos,parse,ner,lemma,depparse',
                'gazetteer': '/scr/nlp/data/machine-reading/Machine_Reading_P1_Reading_Task_V2.0/data/SportsDomain/NFLScoring_UseCase/NFLgazetteer.txt',
                'outputFormat': 'json',
            })
            if isinstance(corenlpres, basestring):
                print corenlpres
                corenlpres = corenlpserver.annotate(s.text.encode("utf8"), properties={
                'ssplit.eolonly': True,
                # 'annotators': 'tokenize,ssplit,pos,depparse,parse',
                'nfl.gazetteer': '/scr/nlp/data/machine-reading/Machine_Reading_P1_Reading_Task_V2.0/data/SportsDomain/NFLScoring_UseCase/NFLgazetteer.txt',
                'annotators': 'tokenize,ssplit,pos,ner,lemma',
                'outputFormat': 'json',
            })
            s.process_corenlp_sentence(corenlpres)


    def tag_chemdner_entity(self, start, end, subtype, **kwargs):
        """
        Create an CHEMDNER entity relative to this document.
        :param start: Start index of entity
        :param end: End index of entity
        :param subtype: Subtype of CHEMDNER entity
        :param kwargs: Extra stuff like the text
        :return:
        """
        doct = kwargs.get("doct")
        if doct == "T": # If it's in the title, we already know the sentence (it's the first)
            self.sentences[0].tag_entity(start, end, subtype, **kwargs)
        else: # we have to find the sentence
            found = False
            totalchars = 0
            for s in self.sentences[1:]:
                if totalchars <= start and totalchars + len(s.text) >= end:  # entity is in this sentence
                    s.tag_entity(start-totalchars, end-totalchars, subtype,
                                 totalchars=totalchars, **kwargs)
                    # print "found entity on sentence %s" % s.sid
                    found = True
                    break

                totalchars += len(s.text)
                totalchars = self.get_space_between_sentences(totalchars)
            if not found:
                print "could not find sentence for %s:%s on %s!" % (start,
                                                                       end, self.did)
                # sys.exit()

    def add_relation(self, entity1, entity2, subtype, relation, source="goldstandard", **kwargs):
        if self.pairs.pairs:
            pid = self.did + ".p" + str(len(self.pairs.pairs))
        else:
            pid = self.did + ".p0"
        between_text = self.text[entity1.dend:entity2.start]
        logging.info("adding {}:{}=>{}".format(pid, entity1.text.encode("utf8"), entity2.text.encode("utf8")))
        # print between_text
        if subtype == "tlink":
            pair = TLink(entity1, entity2, relation=relation, original_id=kwargs.get("original_id"),
                                     did=self.did, pid=pid, rtype=subtype, between_text=between_text)
        else:
            pair = Pair((entity1, entity2), subtype, did=self.did, pid=pid, original_id=kwargs.get("original_id"), between_text=between_text)
        self.pairs.add_pair(pair, source)
        return pair

    def get_space_between_sentences(self, totalchars):
        """
        When the sentences are split, the whitespace between each sentence is not preserved, so we need to get it back
        :param totalchars: offset of the end of sentence
        :return: Index where the next sentence starts
        """
        while totalchars < len(self.text) and self.text[totalchars].isspace():
            totalchars += 1
        return totalchars

    def get_unique_results(self, source, ths, rules, mode):
        entries = set()
        for s in self.sentences:
            if s.entities:
                if mode == "ner":
                    sentence_entries = s.entities.get_unique_entities(source, ths, rules)
                elif mode == "re":
                    sentence_entries = s.entities.get_unique_relations(source)
                entries.update(sentence_entries)
        return entries

    def write_chemdner_results(self, source, outfile, ths={"chebi":0.0}, rules=[]):
        lines = []
        totalentities = 0
        for s in self.sentences:
            # print "processing", s.sid, "with", len(s.entities.elist[source]), "entities"
            if s.entities:
                res = s.entities.write_chemdner_results(source, outfile, ths, rules, totalentities+1)
                lines += res[0]
                totalentities = res[1]
        return lines

    def write_bioc_results(self, parent, source, ths={}):
        bioc_document = ET.SubElement(parent, "document")
        bioc_id = ET.SubElement(bioc_document, "id")
        bioc_id.text = self.did

        bioc_title_passage = ET.SubElement(bioc_document, "passage")
        bioc_title_info = ET.SubElement(bioc_title_passage, "infon", {"key":"type"})
        bioc_title_info.text = "title"
        bioc_title_offset = ET.SubElement(bioc_title_passage, "offset")
        bioc_title_offset.text = str(0)
        bioc_title = self.sentences[0].write_bioc_results(bioc_title_passage, source)

        bioc_abstract_passage = ET.SubElement(bioc_document, "passage")
        bioc_abstract_info = ET.SubElement(bioc_abstract_passage, "infon", {"key":"type"})
        bioc_abstract_info.text = "abstract"
        bioc_abstract_offset = ET.SubElement(bioc_title_passage, "offset")
        bioc_abstract_offset.text = str(len(self.sentences[0].text) + 1)
        for i, sentence in enumerate(self.sentences[1:]):
            bioc_sentence = sentence.write_bioc_results(bioc_abstract_passage, source)
        return bioc_document

    def get_dic(self, source, ths={}):
        dic = {"title":{}, "abstract":{}}
        dic = {"abstract":{}}
        # dic["title"]["offset"] = "0"
        # dic["title"]["sentences"] = self.sentences[0].get_dic(source)

        dic["abstract"]["offset"] = str(len(self.sentences[0].text) + 1)
        dic["abstract"]["sentences"] = []
        for i, sentence in enumerate(self.sentences[1:]):
            dic["abstract"]["sentences"].append(sentence.get_dic(source))
        return dic

    def get_sentence(self, sid):
        """
        Get the sentence by sentence ID
        :param sid: sentence ID
        :return: the sentence object if it exists
        """
        for s in self.sentences:
            # logging.debug([(t.start, t.end) for t in s.tokens])
            if s.sid == sid:
                # logging.debug("found sid: {}".format(sid))
                return s
        return None

    def find_sentence_containing(self, start, end, chemdner=True):
        """
            Find the sentence between start and end. If chemdner, do not consider the first sentence, which
            is the title.
        """
        if chemdner:
            firstsent = 1
        else:
            firstsent = 0
        for i, s in enumerate(self.sentences[firstsent:]):
            if len(s.tokens) == 0:
                logging.debug("sentence without tokens: {} {}".format(s.sid, s.text))
                continue
            if s.tokens[0].dstart <= start and s.tokens[-1].dend >= end:
                # print "found it!"
                return s
        for s in self.sentences:
            print s.tokens[0].dstart <= start, s.tokens[-1].dend >= end, s.tokens[0].dstart, s.tokens[-1].dend, s.text
        return None

    def get_offsets(self, esource, ths, rules, off_list=None):
        #print esource

        offsets = []
        for s in self.sentences:
            #print s.text
            offies = gazette.easy_search_terms(s, esource, ths, rules, off_list)
            if len(offies) == 1:
                offsets += offies #Check it doesn't affect normal results
            else:
                if s.entities:                   
                    offsets += s.entities.get_offsets2(esource, ths, rules)
                    offsets += offies


        return list(set(offsets))


    def get_entity(self, eid, source="goldstandard"):
        for sentence in self.sentences:
            for e in sentence.entities.elist[source]:
                if e.eid == eid:
                   return e
        print "no entity found for eid {}".format(eid)
        return None

    def get_entities(self, source):
        entities = []
        for s in self.sentences:
            if source in s.entities.elist:
                for e in s.entities.elist[source]:
                    entities.append(e)
        return entities
