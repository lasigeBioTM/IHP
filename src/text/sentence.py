from __future__ import unicode_literals
import logging
import socket
import sys
from xml.etree import ElementTree as ET
import re
import pprint
from classification.ner.stanfordner import stanford_coding
from text.protein_entity import ProteinEntity

from token2 import Token2
from entity import Entities
from classification.ner.simpletagger import create_entity
from text.pair import Pair, Pairs
from classification.rext import ddi_kernels
from classification.rext import relations
from text.chemical_entity import ChemicalEntity
from text.mirna_entity import MirnaEntity
from text.event_entity import EventEntity
from text.time_entity import TimeEntity
from text.tlink import TLink

pp = pprint.PrettyPrinter(indent=2)
class Sentence(object):
    """Sentence from a document, to be annotated"""
    def __init__(self, text, offset=0, **kwargs):
        self.text = text
        self.sid = kwargs.get("sid")
        self.did = kwargs.get("did")
        self.entities = Entities(sid=self.sid, did=self.did)
        self.offset = offset
        self.pairs = Pairs()
        self.parsetree = None
        self.depparse = None
        self.tokens = []
        self.regex_tokens = re.compile(r'(-|/|\\|\+|\.|\w+)')

    def tokenize_words(self):
        pass


    def process_corenlp_sentence(self, corenlpres):

        """
        Process the results obtained with CoreNLP for this sentence
        :param corenlpres:
        :return:
        """
        # self.sentences = []
        if len(corenlpres['sentences']) > 1:
            print self.text
            sys.exit("Number of sentences from CoreNLP is not 1.")
        if len(corenlpres['sentences']) == 0:
            self.tokens = []
            self.create_newtoken("", {})
            logging.debug("no sentences")
            logging.debug(self.text)
            return
        sentence = corenlpres['sentences'][0]
        #logging.debug(str(sentence.keys()))
        #print "sentence", self.text.encode("utf8")
        #print "parse", pp.pprint(sentence["parse"])
        #print "basic", pp.pprint(sentence["basic-dependencies"])
        #print "collapsed", pp.pprint(sentence["collapsed-dependencies"])
        #print "ccprocessed", pp.pprint(sentence["collapsed-ccprocessed-dependencies"])
        self.parsetree = sentence.get('parse')
        self.depparse = sentence.get('basic-dependencies')
        for t in sentence['tokens']:
            # print t[0]
            if t["word"]:
                # TODO: specific rules for each corpus
                #if ""
                token_seq = self.regex_tokens.split(t["word"])#, flags=re.U)
                #token_seq = rext.split(r'(\w+)(/|\\|\+|\.)(\w+)', t[0])
                #token_seq = [t[0]]
                # print t[0], token_seq
                if len(token_seq) > 3 and t["word"] not in stanford_coding.keys():
                    # logging.info("{}: {}".format(t["word"], "&".join(token_seq)))
                    for its, ts in enumerate(token_seq):
                        if ts.strip() != "":
                            charoffset_begin = int(t["characterOffsetBegin"])
                            if token_seq[:its]: # not the first token
                                charoffset_begin += sum([len(x) for x in token_seq[:its]])
                            # charoffset_begin += its
                            charoffset_end = len(ts) + charoffset_begin
                            #logging.info(str(charoffset_begin) + ":" + str(charoffset_end))
                            ts_props = {"characterOffsetBegin": charoffset_begin,
                                        "characterOffsetEnd": charoffset_end,
                                        "pos": t["pos"],
                                        "ner": t["ner"],
                                        "lemma": t["lemma"][charoffset_begin:charoffset_end]}
                            self.create_newtoken(ts, ts_props)

                else:
                    self.create_newtoken(t["word"], t)

    def create_newtoken(self, text, props):
        newtoken = Token2(text, order=len(self.tokens))
        try:
            newtoken.start = int(props["characterOffsetBegin"])
            newtoken.dstart = newtoken.start + self.offset
            newtoken.end = int(props["characterOffsetEnd"])
            newtoken.dend = newtoken.end + self.offset
            newtoken.pos = props["pos"]
            newtoken.tag = props["ner"]
            newtoken.lemma = props["lemma"]
            # newtoken.stem = porter.stem_word(newtoken.text)
            newtoken.tid = self.sid + ".t" + str(len(self.tokens))
            self.tokens.append(newtoken)
            # print "|{}| <=> |{}|".format(text, self.text[newtoken.start:newtoken.end])
        except KeyError:
            logging.debug("error: text={} props={}".format(text, props))
            return None
        # logging.debug(newtoken.text)
        return newtoken

    def add_relation(self, entity1, entity2, subtype, source="goldstandard", **kwargs):
        if self.pairs.pairs.get(source):
            pid = self.sid + ".p" + str(len(self.pairs.pairs[source]))
        else:
            pid = self.sid + ".p0"
        if subtype == "tlink":
            p = TLink(entity1, entity2, original_id=kwargs.get("original_id"),
                                     did=self.did, pid=pid, rtype=subtype)
        else:
            p = Pair((entity1, entity2), subtype)
        self.pairs.add_pair(p, source)
        return p

    def exclude_entity(self, start, end, source):
        """
        Exclude all entities matching start-end relative to sentence
        :param start:
        :param end:
        """
        to_delete = []
        for e in self.entities.elist[source]:
            if e.start == start and e.end == end:
                to_delete.append(e)
                for t in e.tokens:
                    tagkeys = t.tags.keys()
                    for tag in tagkeys:
                        if tag.startswith(source):
                            del t.tags[tag]
        for e in to_delete:
            #print "removing {}".format(e)
            self.entities.elist[source].remove(e)
            #print [(ee.start, ee.end) for ee in self.entities.elist[source]]


    def tag_entity(self, start, end, etype, entity=None, source="goldstandard", exclude=None, **kwargs):
        """Find the tokens that match this entity. start and end are relative to the sentence.
           Totalchars is the offset of the sentence on the document."""
        tlist = []
        # print self.tokens
        nextword = ""
        for t in self.tokens:
            # discard tokens that intersect the entity for now
            # print t.start, t.end, t.text
            if t.start >= start and t.end <= end:
                tlist.append(t)
            elif (t.start == start and t.end > end) or (t.start < start and t.end == end):
                tlist.append(t)
                break
            elif t.start == end+1:
                nextword = t.text
            exclude_list = []
            if exclude is not None:
                for t in tlist:
                    for e in exclude:
                        if t.start >= e[0] and t.end <= e[1]-1:
                            exclude_list.append(t.tid)
            tlist = [t for t in tlist if t.tid not in exclude_list]
        if tlist:
            if exclude is not None:
                newtext = self.text[tlist[0].start:exclude[0][0]]
                #print self.text[exclude[0][0]:exclude[0][1]], exclude
                last_exclude = exclude[0]
                for e in exclude[1:]:
                    if not self.text[e[1]].isspace() and not newtext[-1].isspace():
                        newtext += " "
                    newtext += self.text[last_exclude[1]:e[0]]
                    last_exclude = e
                if not self.text[exclude[-1][1]].isspace() and not newtext[-1].isspace():
                    newtext += " "
                newtext += self.text[exclude[-1][1]:tlist[-1].end]
                # self.text[exclude[1]:tlist[-1].end]
            else:
                newtext = self.text[tlist[0].start:tlist[-1].end]
            if entity:
                entity.text = newtext
            if "text" in kwargs and newtext != kwargs["text"]:
                if newtext not in kwargs["text"] and kwargs["text"] not in newtext:
                    logging.info("text does not match: {}=>{}".format(newtext, kwargs["text"]))
                    print exclude, self.text[tlist[0].start:tlist[-1].end]
                    print self.text[tlist[0].start:exclude[0][0]]
                    print self.text[exclude[0][0]:exclude[0][1]]
                    print self.text[exclude[0][1]:tlist[-1].end]

                    # return None
                else:
                    logging.info("diferent text:|system {} {} |{}|=>|{}| {} {} input|{} {}".format(tlist[0].start, tlist[-1].end, newtext, kwargs["text"],
                                 start, end, self.sid, self.text))
                    # print exclude, self.text[tlist[0].start:tlist[-1].end]
            #     print "tokens found:", [t.text for t in tlist]
                    # sys.exit()
            # else:
            # print "found the tokens!", start, end, kwargs["text"], self.sid

            if self.entities.elist.get(source):
                eid = self.sid + ".e" + str(len(self.entities.elist[source]))
            else:
                eid = self.sid + ".e0"
            if entity is None:
                if "text" in kwargs:
                    newtext = kwargs["text"]
                entity = create_entity(tlist, self.sid, did=self.did, text=newtext, score=kwargs.get("score"),
                                       etype=etype, eid=eid, subtype=kwargs.get("subtype"),
                                       original_id=kwargs.get("original_id"), nextword=nextword)
            self.entities.add_entity(entity, source)
            self.label_tokens(tlist, source, etype)
            #logging.debug("added {} to {}, now with {} entities".format(newtext, self.sid,
            #                                                                 len(self.entities.elist[source])))
            return eid
        else:
            logging.info("no tokens found:")
            logging.info("{} {} {} {}".format(self.sid, start, end, kwargs.get("text")))
            logging.info(str([(t.start, t.end, t.text) for t in self.tokens]))

    def label_tokens(self, tlist, source, etype):
        if len(tlist) == 1:
            tlist[0].tags[source] = "single"
            tlist[0].tags[source + "_subtype"] = etype
            tlist[0].tags[source + "_" + etype] = "single"
        else:
            for t in range(len(tlist)):
                if t == 0:
                    tlist[t].tags[source] = "start"
                    tlist[t].tags[source + "_" + etype] = "start"
                    tlist[t].tags[source + "_subtype"] = etype
                elif t == len(tlist) - 1:
                    tlist[t].tags[source] = "end"
                    tlist[t].tags[source + "_" + etype] = "end"
                    tlist[t].tags[source + "_subtype"] = etype
                else:
                    tlist[t].tags[source] = "middle"
                    tlist[t].tags[source + "_" + etype] = "middle"
                    tlist[t].tags[source + "_subtype"] = etype
        #logging.debug([t.tags for t in self.tokens])

    def write_bioc_results(self, parent, source):
        bioc_sentence = ET.SubElement(parent, "sentence")
        bioc_sentence_offset = ET.SubElement(bioc_sentence, "offset")
        bioc_sentence_offset.text = str(self.tokens[0].dstart)
        bioc_sentence_text = ET.SubElement(bioc_sentence, "text")
        bioc_sentence_text.text = self.text

        if source in self.entities.elist:
            for entity in self.entities.elist[source]:
                bioc_annotation = entity.write_bioc_annotation(bioc_sentence)
        return bioc_sentence

    def get_dic(self, source):
        dic = {}
        dic["id"] = self.sid
        dic["offset"] = str(self.tokens[0].dstart)
        dic["text"] = self.text
        dic["entities"] = []
        if source in self.entities.elist:
            for entity in self.entities.elist[source]:
                dic["entities"].append(entity.get_dic())
            dic["entities"] = sorted(dic["entities"], key=lambda k: k['offset'])
            for ei, e in enumerate(dic["entities"]):
                e["eid"] = self.sid + ".e{}".format(ei)
        dic["pairs"] = self.pairs.get_dic()
        return dic

    def find_tokens(self, text, start, end, count, relativeto="doc"):
        candidates = []
        for t in self.tokens:
            if t.text == text:
                print t.text, text
                candidates.append(t)
        print text, candidates
        if len(candidates) == 0:
            print "could not find tokens!"
        elif len(candidates) == 1:
            return candidates
        elif len(candidates)-1 > count:
            candidates[count]
        """else:
            dist = []
            for c in candidates:
                if relativeto == "doc":
                    d = c.dstart
                else:
                    d = c.start
                dist.append(abs(d-start))
            return [candidates[dist.index(min(dist))]]"""

    def find_tokens_between(self, start, end, relativeto="doc"):
        """Return list of tokens between offsets. Use relativeto to consider doc indexes or
           sentence indexes."""
        foundtokens = []
        for t in self.tokens:
            if relativeto.startswith("doc") and t.dstart >= start and t.dend <= end:
                foundtokens.append(t)
            elif relativeto.startswith("sent") and t.start >= start and t.end <= end:
                foundtokens.append(t)
        return foundtokens

    def test_relations(self, pairs, basemodel, classifiers=[relations.SLK_PRED, relations.SST_PRED],
                       tag="", backup=False, printstd=False):
        #data =  ddi_train_slk.model, ddi_train_sst.model
        tempfiles = []

        if relations.SLK_PRED in classifiers:
            logging.info("**Testing SLK classifier %s ..." % (tag,))
            #testpairdic = ddi_kernels.fromddiDic(testdocs)
            ddi_kernels.generatejSREdata(pairs, self, basemodel, tag + "ddi_test_jsre.txt")
            ddi_kernels.testjSRE(tag + "ddi_test_jsre.txt", tag + "ddi_test_result.txt",
                                 model=tag + "all_ddi_train_slk.model")
            self.pairs.pairs = ddi_kernels.getjSREPredicitons(tag + "ddi_test_jsre.txt", tag + "ddi_test_result.txt",
                                                      self.pairs.pairs)
            tempfiles.append(ddi_kernels.basedir + tag + "ddi_test_jsre.txt")
            tempfiles.append(ddi_kernels.basedir + tag + "ddi_test_result.txt")

        if relations.SST_PRED in classifiers:
            logging.info("****Testing SST classifier %s ..." % (tag,))
            self.pairs.pairs = ddi_kernels.testSVMTK(self, self.pairs.pairs, pairs,
                                             model=tag + "all_ddi_train_sst.model", tag=tag)
        for p in self.pairs.pairs:
            for r in self.pairs.pairs[p].recognized_by:
                if self.pairs.pairs[p].recognized_by[r] == 1:
                    p.relation = True
        return tempfiles
