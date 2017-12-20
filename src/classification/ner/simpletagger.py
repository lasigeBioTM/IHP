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
import string

#dic = []
#a = open("data/gazette.txt").readlines()
#for x in a:
#    dic.append(x.strip()) 

paths = open("data/paths").readlines()
path_dic = {}
for x in paths:
    y = x.strip().split("\t")
    path_dic[y[1]] = y[0]


lex = ["the", "of", "and", "in", "is", "as", "for", "that", "was", "are", "this", "we", "were", "by", "have", "from", "these", "be", "had", "been", "which", "may", "has", "not", "other"]#, "but", "who", "than"]

nouns = ["syndrome", "patients", "mutations", "gene", "nf2", "chromosome", "cases", "mutation", "families", "deletion", "clinical"]

feature_extractors = {
                    #Linguistic Features
                      "word": lambda x, i: x.tokens[i].text,
                      "lemma": lambda x, i: x.tokens[i].lemma,
                      "postag": lambda x, i: x.tokens[i].pos,

                    # # #Orthigraphic Features
                      "case": lambda x, i: word_case(x.tokens[i].text),
                      "hasnumber": lambda x, i: str(any(c.isdigit() for c in x.tokens[i].text)),  
                      "hasdash": lambda x, i: contains(x, i, 0, "-"),
                      "hasquote": lambda x, i: contains(x, i, 0, "'"),
                      "hasquote2": lambda x, i: contains(x, i, 0, '"'),
                      "hasparen1": lambda x, i: contains(x, i, 0, "("),
                      "hasparen2": lambda x, i: contains(x, i, 0, ")"),
                      "hasbrack1": lambda x, i: contains(x, i, 0, "["),
                      "hasbrack2": lambda x, i: contains(x, i, 0, "]"),
                      "hasslash": lambda x, i: contains(x, i, 0, '/'),

                    # # #Morphological Features - Maybe find specific affixes
                      "prefix2": lambda x, i: x.tokens[i].text[:2],
                      "prefix3": lambda x, i: x.tokens[i].text[:3],
                      "suffix1": lambda x, i: x.tokens[i].text[-1:],
                      "suffix2": lambda x, i: x.tokens[i].text[-2:],
                      "suffix3": lambda x, i: x.tokens[i].text[-3:],
                      "suffix4": lambda x, i: x.tokens[i].text[-4:],

                      "wordshape": lambda x, i: wordshape(x,i),               
                    
                      "prevbigram": lambda x, i: prevbigram(x, i),

                    # # #Context Features
                      "prevlemma1": lambda x, i: prev_lemma(x,i,1),
                      "nextlemma1": lambda x, i: next_lemma(x,i,1),
                      "prevlemma2": lambda x, i: prev_lemma(x,i,2),
                      "nextlemma2": lambda x, i: next_lemma(x,i,2),

                      "prevpostag2": lambda x, i: prev_pos(x,i,1),
                      "nextpostag2": lambda x, i: next_pos(x,i,1),
                      "prevpostag3": lambda x, i: prev_pos(x,i,2),
                      "nextpostag3": lambda x, i: next_pos(x,i,2),  
                      "nextpostag4": lambda x, i: next_pos(x,i,3),
                      "nextpostag4": lambda x, i: next_pos(x,i,3),
                      "prevpostag5": lambda x, i: prev_pos(x,i,4),
                      "nextpostag5": lambda x, i: next_pos(x,i,4),        

                      "prevprefix1": lambda x, i: prev_prefix(x, i, 0, 1), 
                      "nextprefix1": lambda x, i: next_prefix(x, i, 0, 1),
                      "prevprefix2": lambda x, i: prev_prefix(x, i, 0, 2), 
                      "nextprefix2": lambda x, i: next_prefix(x, i, 0, 2), 
                      "prevprefix3": lambda x, i: prev_prefix(x, i, 0, 3), 
                      "nextprefix3": lambda x, i: next_prefix(x, i, 0, 3),     
                      "prev2prefix1": lambda x, i: prev_prefix(x, i, 1, 1), 
                      "next2prefix1": lambda x, i: next_prefix(x, i, 1, 1), 
                      "prev2prefix2": lambda x, i: prev_prefix(x, i, 1, 2), 
                      "next2prefix2": lambda x, i: next_prefix(x, i, 1, 2), 
                      "prev2prefix3": lambda x, i: prev_prefix(x, i, 1, 3), 
                      "next2prefix3": lambda x, i: next_prefix(x, i, 1, 3), 
                      "prev2suffix1": lambda x, i: prev_suffix(x, i, 1, 1), 
                      "next2suffix1": lambda x, i: next_suffix(x, i, 1, 1), 
                      "prev2suffix2": lambda x, i: prev_suffix(x, i, 1, 2), 
                      "next2suffix2": lambda x, i: next_suffix(x, i, 1, 2), 
                      "prev2suffix3": lambda x, i: prev_suffix(x, i, 1, 3), 
                      "next2suffix3": lambda x, i: next_suffix(x, i, 1, 3), 
                      "prev2suffix4": lambda x, i: prev_suffix(x, i, 1, 4), 
                      "next2suffix4": lambda x, i: next_suffix(x, i, 1, 4), 

                      "prevwordshape": lambda x, i: prev_wordshape(x, i, 1),
                      "nextwordshape": lambda x, i: next_wordshape(x, i, 1),
                      "prevwordshape2": lambda x, i: prev_wordshape(x, i, 2),
                      "nextwordshape2": lambda x, i: next_wordshape(x, i, 2),

                    #  #Lexicon Features - Works better with separate features not using lexicon.
                      "stopwords": lambda x, i: lexicon(x, lex, i, 0),
                      "stopwords_prev1": lambda x, i: lexicon(x, lex, i, -1),
                      "stopwords_next1": lambda x, i: lexicon(x, lex, i, 1),
                      "stopwords_prev2": lambda x, i: lexicon(x, lex, i, -2),
                      "stopwords_next2": lambda x, i: lexicon(x, lex, i, 2),
                      "stopwords_prev3": lambda x, i: lexicon(x, lex, i, -3),
                      "stopwords_next3": lambda x, i: lexicon(x, lex, i, +3),
                      "stopwords_prev4": lambda x, i: lexicon(x, lex, i, -4),
                      "stopwords_next4": lambda x, i: lexicon(x, lex, i, +4),

                    #  #Other Features
                      "brown_cluster": lambda x, i: brown_cluster(x, i, 0),
                      "length": lambda x, i: lengthclass(x, i),
                                             }

def lengthclass(sentence, i):
    if len(sentence.tokens[i].text) <= 3:
        return "A"
    if len(sentence.tokens[i].text) > 3 and len(sentence.tokens[i].text)  <= 6:
        return "B"
    if len(sentence.tokens[i].text) > 6:
        return "C"

def presence(sentence, i, range):
    for word in go_words:
        if word in sentence.text:
            t_flag = False
            start = i-range
            end = i+range
            if start < 0:
                start = 0
            if end > len(sentence.tokens):
                end = len(sentence.tokens)
            for t in sentence.tokens[i-range:i+range]:
                if word in t.text:
                    t_flag = True
            if t_flag:
                return "1"
            else:
                return "2"
        else:
            return "0"

def orthographic(sentence, i, punctuation_type):
    if i >= len(sentence.tokens) - 1:
        return "EOS"
    elif i < 0:
        return "BOS"
    elif sentence.tokens[i].text == punctuation_type:
        return "1"
    else:
        return "0"

def capitalized(sentence,i):
    if sentence.tokens[i].text.istitle():
        return "1"
    else:
        return "0"

def affixes(sentence, i, atype, string):
    p_len = len(string)
    if atype == "prefix":
        if sentence.tokens[i].text[:p_len] == string:
            return "1"
        else:
          return "0"
    if atype == "suffix":
        if sentence.tokens[i].text[-(p_len):] == string:
            return "1"
        else:
            return "0"

def lowercase(sentence,i):
    if sentence.tokens[i].text.islower():
        return "1"
    else:
        return "0"

def uppercase(sentence,i):
    if sentence.tokens[i].text.isupper():
        return "1"
    else:
        return "0"

def mixcase(sentence, i):
    if not sentence.tokens[i].text.isupper() and not sentence.tokens[i].text.islower():
        return "1"
    else:
        return "0"

def singlechar(sentence, i):
    if len(sentence.tokens[i].text) == 1 and sentence.tokens[i].text in string.letters:
        return "0"
    else:
        return "1" 

def singledigit(sentence, i):
    if len(sentence.tokens[i].text) == 1 and sentence.tokens[i].text in string.digits:
        return "0"
    else:
        return "1" 

def doubledigit(sentence, i):
    if (len(sentence.tokens[i].text) == 2 and
         sentence.tokens[i].text[0] in string.digits and
         sentence.tokens[i].text[1] in string.digits):
        return "0"
    else:
        return "1" 

def alphanumeric(sentence, i):
    an_flag = True
    for x in sentence.tokens[i].text:
        if x not in string.digits+string.letters:
            an_flag = False
    if an_flag:
        return "1"
    else:
        return "0"

def roman(sentence, i):
    roman_flag = True
    for x in sentence.tokens[i].text:
        if x not in "XLCDMIV":
            roman_flag = False
    if roman_flag:
        return "1"
    else:
        return "0"

def plural(sentence, i):
    if sentence.tokens[i].pos == "NN":
        return "SING"
    elif sentence.tokens[i].pos == "NNS":
        return "PLUR"
    else:
        return "NONE"

def contains(sentence, i, j, string):
    if i >= len(sentence.tokens) - 1:
        return "EOS"
    if i < 0:
        return "BOS"    
    if string in sentence.tokens[i].text:
        return "1"
    else:
        return "0"
def lexicon(sentence, list, i, j):
    if i+j >= len(sentence.tokens) - 1:
        return "EOS"
    if i-j <= 0:
        return "BOS"
    if sentence.tokens[i].text in list:
        return sentence.tokens[i+j].text.lower()
    else:
        return "NON"

def prevbigram(sentence, i):
    if i-1 <= 0 and i <=0:
        return str(["BOS", "BOS"])
    if i-1 == 0:
        return str(["BOS", sentence.tokens[i].text])
    else:
        return str([sentence.tokens[i-1].text, sentence.tokens[i].text]) 

def nextbigram(sentence, i):
    if i+1 >= len(sentence.tokens) - 1:
        return str(["BOS", "BOS"])
    if i+1 == len(sentence.tokens) - 1:
        return str([sentence.tokens[i].text, "BOS"])
    else:
        return str([sentence.tokens[i].text, sentence.tokens[i+1].text]) 

def prevtrigram(sentence, i):
    if i-2 <= 0 and i <=0:
        return str(["BOS", "BOS", "BOS"])
    if i-1 == 0:
        return str(["BOS", "BOS", sentence.tokens[i].text])
    if i-2 == 0:
        return str(["BOS", sentence.tokens[i-1].text, sentence.tokens[i].text])
    else:
        return str([sentence.tokens[i-2].text, sentence.tokens[i-1].text, sentence.tokens[i].text]) 

def nexttrigram(sentence, i):
    if i+2 >= len(sentence.tokens) - 1:
        return str(["BOS", "BOS", "BOS"])
    if i+1 >= len(sentence.tokens) - 1:
        return str([sentence.tokens[i].text, "BOS", "BOS"])
    if i+2 == len(sentence.tokens) - 1:
        return str([sentence.tokens[i].text, sentence.tokens[i+1].text, "BOS"])
    else:
        return str([sentence.tokens[i].text, sentence.tokens[i+1].text, sentence.tokens[i+2].text]) 

def brown_cluster(sentence, i, j):
    if i+j >= len(sentence.tokens) - 1:
        return "EOS"
    if i-j <= 0:
        return "BOS"
    try:
        return path_dic[sentence.tokens[i+j].text]
    except KeyError:
        return "NON"

def brown_cluster1(sentence, i):
    if i >= len(sentence.tokens) - 1:
        return "EOS"
    if i <= 0:
        return "BOS"
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


def wordshape(sentence, i):
    word = sentence.tokens[i].text
    final = ""
    final2 = ""
    for z in range(len(word)):
        if word[z] in string.ascii_letters:
            if word == word.upper():
                final += "X"
            if word == word.lower():
                final += "x"
        if word in string.digits:
            final += "D"
        if word in string.punctuation:
            final += word
    count = 1
    for j in range(len(final)-1):
        if final[j+1] == final[j]:
            count += 1
        else:
            final2 += str(count) + final[j]
            count = 1
    if count > 1:
        final2 += str(count) + final[0]

    #print final2, "asasdasd"
    return final2

def prev_wordshape(sentence, i, j):
    if i-j <= 0:
        return "BOS"
    else:
        return wordshape(sentence, i)

def next_wordshape(sentence, i, j):
    if i+j >= len(sentence.tokens) - 1:
        return "EOS"
    else:
        return wordshape(sentence, i)


def next_wordclass(sentence, i, j):
    if i+j >= len(sentence.tokens) - 1:
        return "EOS"
    else:
        return wordclass(sentence.tokens[i+j].text)

def prev_wordclass(sentence, i, j):
    if i-j <= 0:
        return "BOS"
    else:
        return wordclass(sentence.tokens[i-j].text)

def next_simplewordclass(sentence, i, j):
    if i+j >= len(sentence.tokens) - 1:
        return "EOS"
    else:
        return simplewordclass(sentence.tokens[i+j].text)

def prev_simplewordclass(sentence, i, j):
    if i-j <= 0:
        return "BOS"
    else:
        return simplewordclass(sentence.tokens[i-j].text)


def prev_suffix(sentence, i, j, size):
    if i-j <= 0:
        return "BOS"
    else:
        return sentence.tokens[i-j].text[-size:]

def next_suffix(sentence, i, j, size):
    if i+j >= len(sentence.tokens) - 1:
        return "EOS"
    else:
        return sentence.tokens[i+j].text[-size:]

def prev_prefix(sentence, i, j, size):
    if i-j <= 0:
        return "BOS"
    else:
        return sentence.tokens[i-j].text[:size]

def next_prefix(sentence, i, j, size):
    if i+j >= len(sentence.tokens) - 1:
        return "EOS"
    else:
        return sentence.tokens[i+j].text[:size]

def prev_case(sentence, i, j):
    if i-j <= 0:
        return "BOS"
    else:
        return word_case(sentence.tokens[i-j].text)

def next_case(sentence, i, j):
    if i+j >= len(sentence.tokens) - 1:
        return "EOS"
    else:
        return word_case(sentence.tokens[i+j].text)

def prev_lemma(sentence, i, j):
    if i-j <= 0:
        return "BOS"
    else:
        return sentence.tokens[i-j].lemma

def next_lemma(sentence, i, j):
    if i+j >= len(sentence.tokens) - 1:
        return "EOS"
    else:
        return sentence.tokens[i+j].lemma

def prev_pos(sentence, i, j):
    if i-j <= 0:
        return "BOS"
    else:
        return sentence.tokens[i-j].pos

def next_pos(sentence, i, j):
    if i+j >= len(sentence.tokens) - 1:
        return "EOS"
    else:
        return sentence.tokens[i+j].pos

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

