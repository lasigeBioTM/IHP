from __future__ import unicode_literals
import logging


class Pair(object):
    """Relation between two entities from the same sentence"""
    def __init__(self, entities, relation, *args, **kwargs):
        self.sid = kwargs.get("sid")
        self.did = kwargs.get("did")
        self.pid = kwargs.get("pid")
        self.between_text = kwargs.get("between_text")
        self.entities = entities
        self.eids = (entities[0].eid, entities[1].eid)
        self.relation = relation
        self.recognized_by = {}
        self.score = 0


    def get_dic(self):
        dic = {}
        dic["eid1"] = self.eids[0]
        dic["eid2"] = self.eids[0]
        dic["pid"] = self.pid
        dic["relation"] = self.relation

    def validate(self):
        """if " and " == self.between_text:
            logging.info("skipped {}".format(self.between_text))
            return False"""
        return True


class Pairs(object):
    """ List of pairs related to a sentence
    """
    def __init__(self, **kwargs):
        self.pairs = []
        self.sid = kwargs.get("sid")
        self.did = kwargs.get("did")

    def get_dic(self):
        dic = []
        for p in self.pairs:
            dic.append(p.get_dic())
        return dic

    def add_pair(self, pair, psource):
            # logging.debug("created new entry %s for %s" % (esource, self.sid))
        #if entity in self.elist[esource]:
        #    logging.info("Repeated entity! %s", entity.eid)
        # print pair.relation, pair.entities
        pair.recognized_by[psource] = 1
        self.pairs.append(pair)