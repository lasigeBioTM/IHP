import logging

from text.entity import Entity
from config import config


stopwords = set()

class EventEntity(Entity):
    """Chemical entities"""
    def __init__(self, tokens, sid, **kwargs):
        # Entity.__init__(self, kwargs)
        super(EventEntity, self).__init__(tokens, **kwargs)
        self.sid = sid
        self.type = "event"
        self.subtype = kwargs.get("subtype")
        self.original_id = kwargs.get("original_id")

    def get_dic(self):
        dic = super(EventEntity, self).get_dic()
        dic["subtype"] = self.subtype
        return dic

    def validate(self, ths, rules, *args, **kwargs):
        """
        Use rules to validate if the entity was correctly identified
        :param rules:
        :return: True if entity does not fall into any of the rules, False if it does
        """
        if "stopwords" in rules:
            # todo: use regex
            words = self.text.split(" ")
            stop = False
            for s in stopwords:
                if any([s == w.lower() for w in words]):
                    logging.debug("ignored stopword %s" % self.text)
                    stop = True
            if stop:
                return False

        if "paren" in rules:
            if (self.text[-1] == ")" and "(" not in self.text) or (self.text[-1] == "]" and "[" not in self.text) or \
                    (self.text[-1] == "}" and "{" not in self.text):
                logging.debug("parenthesis %s" % self.text)
                self.dend -= 1
                self.end -= 1
                self.text = self.text[:-1]
            if (self.text[0] == "(" and ")" not in self.text) or (self.text[0] == "[" and "]" not in self.text) or \
                    (self.text[0] == "{" and "}" not in self.text):
                logging.debug("parenthesis %s" % self.text)
                self.dstart += 1
                self.start += 1
                self.text = self.text[1:]

        if "hyphen" in rules and "-" in self.text and all([len(t) > 3 for t in self.text.split("-")]):
            logging.debug("ignored hyphen %s" % self.text)
            return False

        #if all filters are 0, do not even check
        if "ssm" in ths and ths["ssm"] != 0 and self.ssm_score < ths["ssm"] and self.text.lower() not in chem_words:
            #logging.debug("filtered %s => %s" % (self.text,  str(self.ssm_score)))
            return False

        if "alpha" in rules:
            alpha = False
            for c in self.text.strip():
                if c.isalpha():
                    alpha = True
                    break
            if not alpha:
                logging.debug("ignored no alpha %s" % self.text)
                return False

        if "dash" in rules and (self.text.startswith("-") or self.text.endswith("-")):
            logging.debug("excluded for -: {}".format(self.text))
            return False
        return True
