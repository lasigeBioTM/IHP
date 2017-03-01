import logging

from mirna_base import MirbaseDB
from text.entity import Entity
from config import config

__author__ = 'Andre'

mirna_stopwords = set(["mediated", "expressing", "deficient", "transfected", "dependent", "family", "specific", "null",
                       "independent", "dependant", "overexpressing", "binding", "targets", "induced"])
                       # "mirna", "mirnas", "mir", "hsa-mir"])

mirna_nextstopwords = set(["inhibitor"])
with open(config.stoplist, 'r') as stopfile:
    for l in stopfile:
        w = l.strip().lower()
        if w not in mirna_stopwords and len(w) > 1:
            mirna_stopwords.add(w)
mirna_stopwords.discard("let")
logging.info("Loading miRbase...")
mirna_graph = MirbaseDB(config.mirbase_path)
mirna_graph.load_graph()
logging.info("done.")

class MirnaEntity(Entity):
    def __init__(self, tokens, sid, *args, **kwargs):
        # Entity.__init__(self, kwargs)
        super(MirnaEntity, self).__init__(tokens, **kwargs)
        self.type = "mirna"
        self.subtype = kwargs.get("subtype")
        self.mirna_acc = None
        self.mirna_name = 0
        self.sid = sid
        self.nextword = kwargs.get("nextword")

    def validate(self, ths, rules, *args, **kwargs):
        """
        Use rules to validate if the entity was correctly identified
        :param rules:
        :return: True if entity does not fall into any of the rules, False if it does
        """
        # logging.debug("using these rules: {}".format(rules))
        # logging.debug("{}=>{}:{}".format(self.text.encode("utf-8"), self.normalized, self.normalized_score))
        words = self.text.split("-")
        '''if len(words) > 2 and len(words[-1]) > 3:
            logging.info("big ending: {}".format(self.text))
            self.text = '-'.join(words[:-1])
            words = words[:-1]
            return False'''
        if "stopwords" in rules:
            if self.text.lower() in ["mirna", "mirnas", "mir", "hsa-mir", "microrna", ]:
                logging.debug("ignored stopword %s" % self.text)
                return False
            stop = False
            for i, w in enumerate(words):
                if w.lower() in mirna_stopwords:
                    logging.debug("ignored stopword %s" % self.text)
                    self.text = '-'.join(words[:i])
                    self.dend -= len(words[i:])
                    self.end -= len(words[i:])
                    # stop = True
        if "nextstopword" in rules:
            if self.nextword in mirna_nextstopwords:
                logging.debug("ignored next stop word: {} {}".format(self.text, self.nextword))
                return False

        # if self.text.startswith("MicroRNA-") or self.text.startswith("microRNA-"):
        #    self.text = "mir-" + "-".join(words[1:])
        """if len(words) > 1 and self.text[-1].isdigit() and self.text[-2].isalpha(): #let-7a1 -> let-7a-1
            self.text = self.text[:-1] + "-" + self.text[-1]
        if len(words) > 1 and words[-1].isdigit() and words[-2].isdigit(): # mir-371-373 -> mir-371
            self.text = "-".join(words[:-1])
        words = self.text.split("-")
        if len(words) > 2 and words[2].isalpha() and words[1].isdigit(): # mir-133-a-1 -> mir-133a-1
            # logging.info(words)
            self.text = words[0] + "-" + words[1] + words[2]
            if len(words) > 3:
                self.text += "-" + '-'.join(words[3:])
            logging.info('-'.join(words) + " -> " + self.text)"""

        return True

    def normalize(self):
        if self.text.isalpha():
            self.normalized = "microrna"
            self.normalized_score = 100
            self.normalized_ref = "text"
        else:
            self.normalized, self.normalized_score= mirna_graph.map_label(self.text)
            self.normalized_ref = "mirbase"
        logging.debug("{}=>{}:{}".format(self.text.encode("utf-8"), self.normalized, self.normalized_score))

