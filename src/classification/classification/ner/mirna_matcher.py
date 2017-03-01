import logging

__author__ = 'Andre'
import re
from matcher import MatcherModel

class MirnaMatcher(MatcherModel):
    """
       Find miRNAs based on a fixed set of expressions.
       Does not need training, since it is based on the fixed nomenclature of miRNAs.
    """
    def __init__(self, path, **kwargs):
        super(MirnaMatcher, self).__init__(path, **kwargs)
        # best prefixes for miRNA corpus
        # self.names = set(["mir", "let", "miR", "hsa", "microRNA", "MicroRNA", "miR", "mir", "miR", "lin", "MiR",
        #                  "miRNA", "hsa-miR", "miRNA", "Let", "pre-miR", "premiR", "Hsa-miR", "Mir", "cel-miR"])
        # best prefixes for miRTex corpus
        self.names = set(["mir", "let", "miR", "hsa", "microRNA", "MicroRNA", "miR", "mir", "miR", "lin", "MiR",
                          "miRNA", "hsa-miR", "miRNA", "Let", "Hsa-miR", "Mir", "cel-miR", "lentivirus-miR", "-"])
        # eventually combine both
        # these expressions may be used to refer to multiple miRNAs
        #self.separators = set(["/",r"\s", r",\s", r"\sand\s", "-", r"\sand\s"])

    def test(self, corpus):
        for n in self.names:
            # logging.info(n)
            # regex explanation:
            # start with (, start of string or whitespace
            # include the prefix and then words or dashes
            # end with whitespace, end of string, dot, comma or )
            # best for miRNA corpus
            self.p.append(re.compile(r"(\(|\A|\s|Ad-|pEGFP-)(" + n + r"[\s-]?\d{1,3}?\w?-?[a-z]?/?\d{0,3}?\w?)(\s|\Z|\.|\*|//|,|-[a-z]{3,}|\))")) # , rext.I))
            # best for miRTex
            #self.p.append(re.compile(r"(\(|\A|\s|\w|)(" + n + r"[\s-]?\d{1,3}?\w?-?[a-z]?\d{0,3}?\w?\*?)(\/|\s|\Z|\.|\*|,|-[a-z]{3,}|\)|\()")) # , rext.I))

            # self.p.append(rext.compile(r"(\(|\A|\s)(" + rext.escape(n) + r"[\w-]*[" + "|".join(self.separators) + r"\w" + r"]*)(\Z|\.|\)|/)"))
        # self.p = [rext.compile(r"(\A|\s)(" + n + r")(\s|\Z|\.)", rext.I) for n in self.names]
        logging.info("testing {} documents".format(len(corpus.documents)))
        logging.debug("with these patterns:")
        for r in self.p:
            logging.debug(r.pattern)
        did_count = 1
        elist = {}
        for did in corpus.documents:
            logging.info("document {0} {1}/{2}".format(did, did_count, len(corpus.documents)))
            for sentence in corpus.documents[did].sentences:
                # sentence.entities.elist["matcher"] = \
                self.tag_sentence(sentence, "mirna")
                if self.path in sentence.entities.elist:
                    for entity in sentence.entities.elist[self.path]:
                        # print entity.start, entity.end
                        elist[entity.eid] = entity
            did_count += 1
        return corpus, elist
