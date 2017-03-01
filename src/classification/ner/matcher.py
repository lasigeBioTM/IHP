import logging
import pickle
import re
from text.offset import partial_overlap_after, partial_overlap_before, contained_by, perfect_overlap, Offsets, Offset, \
    contains


class MatcherModel(object):
    """Model which matches a fixed list of entities to the text"""

    def __init__(self, path, **kwargs):
        self.path = path
        self.names = set()
        self.p = []

    def train(self, corpus):
        for did in corpus.documents:
            for sentence in corpus.documents[did].sentences:
                #print sentence.entities.elist.keys()
                if "goldstandard" in sentence.entities.elist:
                    for entity in sentence.entities.elist["goldstandard"]:
                        if entity.text == "a" or entity.text == "as":
                            print sentence.sid, sentence.text, entity.dstart, entity.dend, entity.text
                            print
                            continue
                        self.names.add(entity.text)
        logging.info("Created set of {} entity names".format(len(self.names)))
        pickle.dump(self.names, open(self.path, "wb"))
        logging.info("saved to {}".format(self.path))

    def train_list(self, listpath):
        with open(listpath, "r") as listfile:
            for l in listfile:
                self.names.add(l.strip().lower())
        logging.info("Created set of {} entity names".format(len(self.names)))
        pickle.dump(self.names, open(self.path, "wb"))
        logging.info("saved to {}".format(self.path))

    def test(self, corpus):
        logging.info("loading names...")
        #self.names = pickle.load(open(self.path, "rb"))
        logging.info("compiling regex...")
        for n in self.names:
            # logging.info(n)
            self.p.append(re.compile(r"(\A|\s)(" + re.escape(n) + r")(\s|\Z|\.|,)", re.I))
        # self.p = [re.compile(r"(\A|\s)(" + n + r")(\s|\Z|\.)", rext.I) for n in self.names]
        logging.info("testing {} documents".format(len(corpus.documents)))
        did_count = 1
        elist = {}
        for did in corpus.documents:
            logging.info("document {0} {1}/{2}".format(did, did_count, len(corpus.documents)))
            for sentence in corpus.documents[did].sentences:
                # sentence.entities.elist["matcher"] = \
                self.tag_sentence(sentence)
                if self.path in sentence.entities.elist:
                    for entity in sentence.entities.elist[self.path]:
                        elist[entity.eid] = entity
            did_count += 1
        return corpus, elist

    def test_alt(self, corpus):
        logging.info("loading names...")
        self.names = pickle.load(open(self.path, "rb"))
        logging.info("compiling regex...")
        self.p = re.compile(r"(\A|\s)(" + "|".join([re.escape(n) for n in self.names]) + r")(\s|\Z|\.)")
        logging.info("testing {} documents".format(len(corpus.documents)))
        did_count = 1
        elist = {}
        for did in corpus.documents:
            logging.info("document {0} {1}/{2}".format(did, did_count, len(corpus.documents)))
            for sentence in corpus.documents[did].sentences:
                self.tag_sentence(sentence)
                for entity in sentence.entities.elist[self.path]:
                    elist[entity.eid] = entity
            did_count += 1
        return corpus, elist

    def tag_sentence(self, sentence, entity_type="entity", offsets=None):
        exclude_this_if = (partial_overlap_after, partial_overlap_before, contained_by, perfect_overlap)
        exclude_others_if = (contains,)
        if not offsets:
            offsets = Offsets()
        for pattern in self.p:
            iterator = pattern.finditer(sentence.text)
            for match in iterator:
                offset = Offset(*match.span(2))
                logging.info(match.group(2))
                toadd, v, overlapping, to_exclude = offsets.add_offset(offset, exclude_this_if, exclude_others_if)
                if toadd:
                    #print sentence.sid, (offset.start,offset.end), [(o.start, o.end) for o in offsets.offsets]
                    sentence.tag_entity(offset.start, offset.end, etype=entity_type, source=self.path)
                    for o in to_exclude:
                        # print "excluding {}-{}".format(o.start,o.end)
                        sentence.exclude_entity(o.start, o.end, self.path)
