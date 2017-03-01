import logging
import re
import MySQLdb
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '../..'))
from config.config import go_conn as db
from config import config
from text.entity import Entity
from text.token2 import Token2

__author__ = 'Andre'
prot_words = set()
prot_stopwords = set(["chromosome", "factor", "conserved", "gene", "anti", "mir", "regulatory", "terminal", "element",
                      "activator", "cell", "box", "transcriptional", "transcription", "growth", "talk", "epithelial",
                      "alpha", "microrna", "chip", "chipseq", "interferons", "tweak", "allele"])
# words that may seem like they are not part of named chemical entities but they are

# words that are never part of chemical entities
with open(config.stoplist, 'r') as stopfile:
    for l in stopfile:
        w = l.strip().lower()
        if w not in prot_words and len(w) > 1:
            prot_stopwords.add(w)


class ProteinEntity(Entity):
    def __init__(self, tokens, sid, *args, **kwargs):
        # Entity.__init__(self, kwargs)
        super(ProteinEntity, self).__init__(tokens, *args, **kwargs)
        self.type = "protein"
        self.subtype = kwargs.get("subtype")
        self.sid = sid

    tf_regex = re.compile(r"\A[A-Z]+\d*\w*\d*\Z")

    def get_dic(self):
        dic = super(ProteinEntity, self).get_dic()
        dic["subtype"] = self.subtype
        dic["ssm_score"] = self.ssm_score
        dic["ssm_entity"] = self.ssm_go_ID
        return dic

    def validate(self, ths, rules, *args, **kwargs):
        """
        Use rules to validate if the entity was correctly identified
        :param rules:
        :return: True if entity does not fall into any of the rules, False if it does
        """
        if "stopwords" in rules:
            words = self.text.split(" ")
            words += self.text.split("-")
            stop = False
            for s in prot_stopwords:
                if any([s == w.lower() for w in words]):
                    logging.debug("ignored stopword %s" % self.text)
                    stop = True
            if stop:
                return False
        if "alpha" in rules and not self.text[0].isalpha():
            logging.debug("not alpha %s" % self.text)
            return False
        if "nwords" in rules:
            words = self.text.split(" ")
            if len(words) > 1:
                return False
        if "codeonly" in rules:
            if self.tf_regex.match(self.text) is None:
                return False
        if "fixdash" in rules:
            self.text = self.text.replace("-", "")
        return True

    def normalize(self):
        term = MySQLdb.escape_string(self.text)
        # adjust - adjust the final score
        match = ()
        cur = db.cursor()
        # synonym
        query = """SELECT DISTINCT t.acc, t.name, s.term_synonym
                       FROM term t, term_synonym s
                       WHERE s.term_synonym LIKE %s and s.term_id = t.id
                       ORDER BY t.ic ASC
                       LIMIT 1;""" # or DESC
            # print "QUERY", query

        cur.execute(query, ("%" + term + "%",))

        res = cur.fetchone()
        if res is not None:
            print res
        else:
            query = """SELECT DISTINCT t.acc, t.name, p.name
                       FROM term t, prot p, prot_GOA_BP a
                       WHERE p.name LIKE %s and p.id = a.prot_id and a.term_id = t.id
                       ORDER BY t.ic ASC
                       LIMIT 1;""" # or DESC
            cur.execute(query, (term,))
            res = cur.fetchone()
            print res

token = Token2("IL-2")
token.start, token.dstart, token.end, token.dend = 0,0,0,0
p = ProteinEntity([token], "", text=sys.argv[1])
p.normalize()