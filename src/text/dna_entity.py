import logging
import re
from config import config
from text.entity import Entity

__author__ = 'Andre'
dna_words = set()
dna_stopwords = set([])
# words that may seem like they are not part of named chemical entities but they are

# words that are never part of chemical entities
with open(config.stoplist, 'r') as stopfile:
    for l in stopfile:
        w = l.strip().lower()
        if w not in dna_words and len(w) > 1:
            dna_stopwords.add(w)


class DNAEntity(Entity):
    def __init__(self, tokens, *args, **kwargs):
        # Entity.__init__(self, kwargs)
        super(DNAEntity, self).__init__(tokens, *args, **kwargs)
        self.type = "dna"
        self.subtype = kwargs.get("subtype")
        # print self.sid

    tf_regex = re.compile(r"\A[A-Z]+\d*\w*\d*\Z")

    def get_dic(self):
        dic = super(DNAEntity, self).get_dic()
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
            for s in dna_stopwords:
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