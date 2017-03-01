import codecs
import logging
import unicodedata
from classification.model import Model
from text.chemical_entity import element_base, ChemicalEntity
from text.chemical_entity import amino_acids
from text.dna_entity import DNAEntity
from text.entity import Entity
#from text.mirna_entity import MirnaEntity
#from text.protein_entity import ProteinEntity
from text.time_entity import TimeEntity
from text.event_entity import EventEntity
from text.hpo_entity import HPOEntity

#dic = []
#a = open("data/gazette.txt").readlines()
#for x in a:
#    dic.append(x.strip()) 

paths = open("data/paths").readlines()
path_dic = {}
for x in paths:
    y = x.strip().split("\t")
    path_dic[y[1]] = y[0]


feature_extractors = {"text": lambda x, i: x.tokens[i].text,
                      "prefix3": lambda x, i: x.tokens[i].text[:3],
                      "prevprefix3": lambda x, i: prev_prefix(x, i, 3),
                      #"nextprefix3": lambda x, i: next_prefix(x, i, 3),
                      #"suffix3": lambda x, i: x.tokens[i].text[-3:],
                      "prevsuffix3": lambda x, i: prev_suffix(x, i, 3),
                      "nextsuffix3": lambda x, i: next_suffix(x, i, 3),
                      "prefix2": lambda x, i: x.tokens[i].text[:2],
                      #"suffix2": lambda x, i: x.tokens[i].text[-2:], #Removes total jaw cysts.
                      #"prefix4": lambda x, i: x.tokens[i].text[:4],
                      #"suffix4": lambda x, i: x.tokens[i].text[-4:],
                      #"hasnumber": lambda x, i: str(any(c.isdigit() for c in x.tokens[i].text)),
                      "case": lambda x, i: word_case(x.tokens[i].text),
                      #"prevcase": lambda x, i: prev_case(x, i),
                      #"nextcase": lambda x, i: next_case(x, i),
                      "lemma": lambda x, i: x.tokens[i].lemma, #Makes go down. Although with next ones go very slightly up.
                      "prevlemma": lambda x, i: prev_lemma(x,i),
                      #"nextlemma": lambda x, i: next_lemma(x,i),
                      "postag": lambda x, i: x.tokens[i].pos,
                      "prevpostag": lambda x, i: prev_pos(x,i),
                      "nextpostag": lambda x, i: next_pos(x,i),
                      "wordclass": lambda x, i: wordclass(x.tokens[i].text),
                      "prevwordclass": lambda x, i: prev_wordclass(x, i),
                      "prevword": lambda x, i: prev_word(x, 1),
                      "prevword2": lambda x, i: prev_word(x, 2),
                      "prevword3": lambda x, i: prev_word(x, 3),
                      "prevword4": lambda x, i: prev_word(x, 4),
                      "prevword5": lambda x, i: prev_word(x, 5),
                      "prevword6": lambda x, i: prev_word(x, 6),
                      "nextword": lambda x, i: prev_word(x, 1),
                      "nextword2": lambda x, i: prev_word(x, 2),
                      "nextword3": lambda x, i: prev_word(x, 3),
                      "nextword4": lambda x, i: prev_word(x, 4),
                      "nextword5": lambda x, i: prev_word(x, 5),
                      "nextword6": lambda x, i: prev_word(x, 6),
                      #"in_dic": lambda x, i: word_in_dictionary(x.tokens[i].text, dic),
                      "prev_word_and": lambda x, i: prev_word_and(x, i),
                      "next_word_and": lambda x, i: next_word_and(x, i),
                      "brown_cluster": lambda x, i: brown_cluster(x, i),
                      #"prev_words": lambda x, i: prev_words(x, i, 2),
                      #"next_words": lambda x, i: next_words(x, i, 6),


                      #"nextwordclass": lambda x, i: next_wordclass(x, i),
                      #"simplewordclass": lambda x, i: simplewordclass(x.tokens[i].text),
                      # "greek": lambda x, i: str(has_greek_symbol(x.tokens[i].text)),
                      # "aminoacid": lambda x, i: str(any(w in amino_acids for w in x.tokens[i].text.split('-'))),
                      # "periodictable": lambda x, i: str(x.tokens[i].text in element_base.keys() or x.tokens[i].text.title() in zip(*element_base.values())[0]), # this should probably be its own function ffs
                      }

def prev_wordclass(sentence, i):
    if i == 0:
        return "BOS"
    else:
        return wordclass(sentence.tokens[i-1].text)


#####
def brown_cluster(sentence, i):
    try:
        return path_dic[sentence.tokens[i].text]
    except KeyError:
        return "NON"

def prev_words(sentence, i, j):
    if i-j <= 0:
        return "BOS"
#    if i+j >= len(sentence.tokens) - 1:
#        return "EOS"
    else:
        words = []
        for p in reversed(range(1,j+1)):
            words.append(sentence.tokens[i-p].text)
        #print str(words)
        return str(words)

def next_words(sentence, i, j):
#    if i-j <= 0:
#        return "BOS"
    if i+j >= len(sentence.tokens) - 1:
        return "EOS"
    else:
        words = []
        for p in reversed(range(1,j+1)):
            words.append(sentence.tokens[i+p].text)
        return str(words)

def prev_word(sentence, i):
    if i <= 0:
        return "BOS"
    if i >= len(sentence.tokens) - 1:
        return "EOS"
    else:
        return sentence.tokens[i-1].text

def next_word(sentence, i):
    if i >= len(sentence.tokens) - 1:
        return "EOS"
    if i <= 0:
        return "BOS"
    else:
        return sentence.tokens[i+1].text

def prev_word_and(sentence, i):
    if i <= 0:
        return "BOS"
    else:
        if sentence.tokens[i-1].text == "and" or sentence.tokens[i-1].text == "or":
            return "1"
        else:
            return "0"

def next_word_and(sentence, i):
    if i >= len(sentence.tokens) - 1:
        return "EOS"
    else:
        if sentence.tokens[i+1].text == "and" or sentence.tokens[i+1].text == "or":
            return "1"
        else:
            return "0"

#####

def next_wordclass(sentence, i):
    if i == len(sentence.tokens) - 1:
        return "EOS"
    else:
        return wordclass(sentence.tokens[i+1].text)

def prev_suffix(sentence, i, size):
    if i == 0:
        return "BOS"
    else:
        return sentence.tokens[i-1].text[-size:]

def next_suffix(sentence, i, size):
    if i == len(sentence.tokens) - 1:
        return "EOS"
    else:
        return sentence.tokens[i+1].text[-size:]

def prev_prefix(sentence, i, size):
    if i == 0:
        return "BOS"
    else:
        return sentence.tokens[i-1].text[:size]

def next_prefix(sentence, i, size):
    if i == len(sentence.tokens) - 1:
        return "EOS"
    else:
        return sentence.tokens[i+1].text[:size]

def prev_case(sentence, i):
    if i == 0:
        return "BOS"
    else:
        return word_case(sentence.tokens[i-1].text)

def next_case(sentence, i):
    if i == len(sentence.tokens) - 1:
        return "EOS"
    else:
        return word_case(sentence.tokens[i+1].text)

def prev_lemma(sentence, i):
    if i == 0:
        return "BOS"
    else:
        return sentence.tokens[i-1].lemma

def next_lemma(sentence, i):
    if i == len(sentence.tokens) - 1:
        return "EOS"
    else:
        return sentence.tokens[i+1].lemma

def prev_pos(sentence, i):
    if i == 0:
        return "BOS"
    else:
        return sentence.tokens[i-1].pos

def next_pos(sentence, i):
    if i == len(sentence.tokens) - 1:
        return "EOS"
    else:
        return sentence.tokens[i+1].pos

def word_case(word):
    if word.islower():
        case = 'LOWERCASE'
    elif word.isupper():
        case = 'UPPERCASE'
    elif word.istitle():
        case = 'TITLECASE'
    else:
        case = 'MIXEDCASE'
    return case


def has_greek_symbol(word):
    for c in word:
        #print c
        try:
            if 'GREEK' in unicodedata.name(c):
                hasgreek = 'HASGREEK'
                return True
        except ValueError:
            return False
    return False


def get_prefix_suffix(word, n):
    #print len(word.decode('utf-8'))
    #if len(word.decode('utf-8')) <= n:
    if len(word) <= n:
        #print "111111"
        #word2 = word.encode('utf-8')
        return word, word
    else:
        #print "22222"
        #return word.decode('utf-8')[:n].encode('utf-8'), word.decode('utf-8')[-n:].encode('utf-8')
        return word[:n], word[-n:]


def wordclass(word):
    wclass = ''
    for c in word:
        if c.isdigit():
            wclass += '0'
        elif c.islower():
            wclass += 'a'
        elif c.isupper():
            wclass += 'A'
        else:
            wclass += 'x'
    return wclass


def simplewordclass(word):
    wclass = '.'
    for c in word:
        if c.isdigit() and wclass[-1] != '0':
            wclass += '0'
        elif c.islower() and wclass[-1] != 'a':
            wclass += 'a'
        elif c.isupper() and wclass[-1] != 'A':
            wclass += 'A'
        elif not c.isdigit() and not c.islower() and not c.isupper() and wclass[-1] != 'x':
            wclass += 'x'
    return wclass[1:]


class SimpleTaggerModel(Model):
    """Model trained with a tagger"""
    def __init__(self, path, etype, **kwargs):
        """
        Generic NER classifier
        :param path: Location of the model file
        :param etype: type of entities classified
        """
        super(SimpleTaggerModel, self).__init__(path, **kwargs)
        self.sids = []
        self.tagger = None
        self.trainer = None
        self.sentences = []
        self.etype = etype

    def load_data(self, corpus, flist, etype="all", mode="train", doctype="all"):
        """
            Load the data from the corpus to the format required by crfsuite.
            Generate the following variables:
                - self.data = list of features for each token for each sentence
                - self.labels = list of labels for each token for each sentence
                - self.sids = list of sentence IDs
                - self.tokens = list of tokens for each sentence
        """
        logging.info("Loading data for type %s" % etype)
        fname = "f" + str(len(flist))
        nsentences = 0
        didx = 0
        savecorpus = False # do not save the corpus if no new features are generated
        for did in corpus.documents:
            if doctype != "all" and doctype not in did:
                continue
            # logging.debug("processing doc %s/%s" % (didx, len(corpus.documents)))
            for si, sentence in enumerate(corpus.documents[did].sentences):
                # skip if no entities in this sentence
                if sentence.sid in corpus.documents[did].invalid_sids:
                    logging.debug("Invalid sentence: {} - {}".format(sentence.sid, sentence.text))
                    continue
                if sentence.sid in corpus.documents[did].title_sids:
                    logging.debug("Title sentence: {} - {}".format(sentence.sid, sentence.text))
                    continue
                if mode == "train" and "goldstandard" not in sentence.entities.elist:
                    # logging.debug("Skipped sentence without entities: {}".format(sentence.sid))
                    continue
                sentencefeatures = []
                sentencelabels = []
                sentencetokens = []
                sentencesubtypes = []
                for i in range(len(sentence.tokens)):
                    if sentence.tokens[i].text:
                        tokensubtype = sentence.tokens[i].tags.get("goldstandard_subtype", "none")
                        # if fname in sentence.tokens[i].features:
                        #     tokenfeatures = sentence.tokens[i].features[fname]
                            #logging.info("loaded features from corpus: %s" % tokenfeatures)
                        #     if etype == "all":
                        #         tokenlabel = sentence.tokens[i].tags.get("goldstandard", "other")
                        #     else:
                        #         tokenlabel = sentence.tokens[i].tags.get("goldstandard_" + type, "other")
                        # else:
                        tokenfeatures, tokenlabel = self.generate_features(sentence, i, flist, etype)
                        # savecorpus = True
                        sentence.tokens[i].features[fname] = tokenfeatures[:]
                        # if tokenlabel != "other":
                        #      logging.debug("%s %s" % (tokenfeatures, tokenlabel))
                        sentencefeatures.append(tokenfeatures)
                        sentencelabels.append(tokenlabel)
                        sentencetokens.append(sentence.tokens[i])
                        sentencesubtypes.append(tokensubtype)
                #logging.info("%s" % set(sentencesubtypes))
                #if subtype == "all" or subtype in sentencesubtypes:
                #logging.debug(sentencesubtypes)
                nsentences += 1
                self.data.append(sentencefeatures)
                self.labels.append(sentencelabels)
                self.sids.append(sentence.sid)
                self.tokens.append(sentencetokens)
                self.subtypes.append(set(sentencesubtypes))
                self.sentences.append(sentence.text)
            didx += 1
        # save data back to corpus to improve performance
        #if subtype == "all" and savecorpus:
        #    corpus.save()
        logging.info("used %s sentences for model %s" % (nsentences, etype))

    def copy_data(self, basemodel, t="all"):
        #logging.debug(self.subtypes)
        if t != "all":
            right_sents = [i for i in range(len(self.subtypes)) if t in self.subtypes[i]]
            #logging.debug(right_sents)
            self.data = [basemodel.data[i] for i in range(len(basemodel.subtypes)) if i in right_sents]
            self.labels = [basemodel.labels[i] for i in range(len(basemodel.subtypes)) if i in right_sents]
            self.sids = [basemodel.sids[i] for i in range(len(basemodel.subtypes)) if i in right_sents]
            self.tokens =  [basemodel.tokens[i] for i in range(len(basemodel.subtypes)) if i in right_sents]
            self.sentences = [basemodel.sentences[i] for i in range(len(basemodel.subtypes)) if i in right_sents]
        else:
            self.data = basemodel.data[:]
            self.labels = basemodel.labels[:]
            self.sids = basemodel.sids
            self.tokens = basemodel.tokens[:]
            self.sentences = basemodel.sentences[:]
        logging.info("copied %s for model %s" % (len(self.data), t))

    def generate_features(self, sentence, i, flist, subtype):
        """
            Features is dictionary mapping of featurename:value.
            Label is the correct label of the token. It is always other if
            the text is not annotated.
        """
        if subtype == "all":
            label = sentence.tokens[i].tags.get("goldstandard", "other")
        else:
            label = sentence.tokens[i].tags.get("goldstandard_" + subtype, "other")
        features = []
        for f in flist:
            if f not in sentence.tokens[i].features:
                fvalue = feature_extractors[f](sentence, i)
                sentence.tokens[i].features[f] = fvalue
            else:
                fvalue = sentence.tokens[i].features[f]
            features.append(f + "=" + fvalue)
        # if label != "other":
        #     logging.debug("{} {}".format(sentence.tokens[i], label))
        #logging.debug(features)
        return features, label

    def save_corpus_to_sbilou(self):
        """
        Saves the data that was loaded into simple tagger format to a file compatible with Stanford NER
        :param entity_type:
        :return:
        """
        logging.info("saving loaded corpus to Stanford NER format...")
        lines = []
        for isent, sentence in enumerate(self.sids):
            for it, l in enumerate(self.labels[isent]):
                if l == "other":
                    label = "O"
                elif l == "start":
                    label = "B-{}".format(self.etype.upper())
                elif l == "end":
                    label = "E-{}".format(self.etype.upper())
                elif l == "middle":
                    label = "I-{}".format(self.etype.upper())
                elif l == "single":
                    label = "S-{}".format(self.etype.upper())
                #label += "_" + entity_type
                try:
                    lines.append("{0}\t{1}\n".format(self.tokens[isent][it].text, label))
                except UnicodeEncodeError: #fml
                    lines.append(u"{0}\t{1}\n".format(self.tokens[isent][it].text, label))
            lines.append("\n")
        with codecs.open("{}.bilou".format(self.path), "w", "utf-8") as output:
            output.write("".join(lines))
        logging.info("done")


def create_entity(tokens, sid, did, text, score, etype, **kwargs):
    """
    Create a new entity based on the type of model
    :param tokens: list of Tokens
    :param sid: ID of the sentence
    :param did: ID of the document
    :param text: string
    :param score:
    :param etype: Type of the entity
    :return: entity
    """
    e = None
    if etype == "chemical":
        e = ChemicalEntity(tokens, sid, text=text, score=score,
                           did=did, eid=kwargs.get("eid"), subtype=kwargs.get("subtype"))
    elif etype == "hpo":
        e = HPOEntity(tokens, sid, text=text, did=did,score=score,
                          eid=kwargs.get("eid"), subtype=kwargs.get("subtype"), nextword=kwargs.get("nextword"))
    elif etype == "mirna":
        e = MirnaEntity(tokens, sid, text=text, did=did, score=score,
                        eid=kwargs.get("eid"), subtype=kwargs.get("subtype"), nextword=kwargs.get("nextword"))
    elif etype == "protein":
        e = ProteinEntity(tokens, sid, text=text, did=did,score=score,
                          eid=kwargs.get("eid"), subtype=kwargs.get("subtype"), nextword=kwargs.get("nextword"))
    elif etype == "dna":
        e = DNAEntity(tokens, sid, text=text, did=did,score=score,
                          eid=kwargs.get("eid"), subtype=kwargs.get("subtype"), nextword=kwargs.get("nextword"))
    elif etype == "event":
         e = EventEntity(tokens, sid, text=text, did=did,score=score,
                         eid=kwargs.get("eid"), subtype=kwargs.get("subtype"), nextword=kwargs.get("nextword"),
                         original_id=kwargs.get("original_id"))
    elif etype in ("timex3", "sectiontime", "doctime"):
         e = TimeEntity(tokens, sid, text=text, did=did,score=score,
                        eid=kwargs.get("eid"), subtype=kwargs.get("subtype"), nextword=kwargs.get("nextword"),
                        original_id=kwargs.get("original_id"))
    else:
        e = Entity(tokens, sid, text=text, did=did,score=score,
                        eid=kwargs.get("eid"), subtype=kwargs.get("subtype"), nextword=kwargs.get("nextword"),
                        original_id=kwargs.get("original_id"), sid=sid)
        e.type = etype
    return e


class BiasModel(SimpleTaggerModel):
    """Model which cheats by using the gold standard tags"""

    def test(self):
        # TODO: return results
        self.predicted = self.labels

