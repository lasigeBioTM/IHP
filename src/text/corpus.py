
import logging
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '../..'))



class Corpus(object):
    """
    Base corpus class
    """
    def __init__(self, corpusdir, **kwargs):
        self.path = corpusdir
        self.documents = kwargs.get("documents", {})
        self.invalid_sections = set()
        self.invalid_sids = set()
        #logging.debug("Created corpus with {} documents".format(len(self.documents)))

    def progress(self, count, total, suffix=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))
        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))

    def save(self, savedir, *args):
        """Save corpus object to a pickle file"""
        # TODO: compare with previous version and ask if it should rewrite
        logging.info("saving corpus...")
        #if not args:
        #    path = "data/" + self.path.split('/')[-1] + ".pickle"
        #else:
        #    path = args[0]
        pickle.dump(self, open(savedir, "wb"))
        logging.info("saved corpus to " + savedir)

    def get_unique_results(self, source, ths, rules, mode):
        allentries = set()
        for d in self.documents:
            if mode == "ner":
                doc_entities = self.documents[d].get_unique_results(source, ths, rules, mode)
                allentries.update(doc_entities)
            elif mode == "re":
                doc_pairs = set()
                # logging.info(len(self.documents[d].pairs.pairs))
                for p in self.documents[d].pairs.pairs:
                    if source in p.recognized_by:
                        doc_pairs.add((d, p.entities[0].text, p.entities[1].text))
                allentries.update(doc_pairs)
        return allentries

    def write_chemdner_results(self, source, outfile, ths={"chebi":0.0}, rules=[]):
        """
        Produce results to be evaluated with the BioCreative CHEMDNER evaluation script
        :param source: Base model path
        :param outfile: Text Results path to be evaluated
        :param ths: Thresholds
        :param rules: Validation rules
        :return:
        """
        lines = []
        cpdlines = []
        max_entities = 0
        for d in self.documents:
            doclines = self.documents[d].write_chemdner_results(source, outfile, ths, rules)
            lines += doclines
            hast = 0
            hasa = 0
            for l in doclines:
                if l[1].startswith("T"):
                    hast += 1
                elif l[1].startswith("A"):
                    hasa += 1
            # print hast, hasa
            cpdlines.append((d, "T", hast))
            if hast > max_entities:
                max_entities = hast
            cpdlines.append((d, "A", hasa))
            if hasa > max_entities:
                max_entities = hasa
            # print max_entities
        return lines, cpdlines, max_entities

    def get_offsets(self, esource, ths, rules):
        """
        Retrieve the offsets of entities found with the models in source to evaluate
        :param esources:
        :return: List of tuple : (did, start, end, text)
        """
        
        offsets = {} # {did1: [(0,5), (10,14)], did2: []...}
        for did in self.documents:
            offsets[did] = self.documents[did].get_offsets(esource, ths, rules)
        offsets_list = []
        for did in offsets:
            for o in offsets[did]:
                offsets_list.append((did, o[0], o[1], o[2]))
                #print did, o[0], o[1], o[2]

        return offsets_list


    def find_chemdner_result(self, res):
        """
            Find the tokens that correspond to a given annotation:
            (did, A/T:start:end)
        """
        did = res[0]
        stype, start, end = res[1].split(":")
        start = int(start)
        end = int(end)
        if stype == 'T':
            sentence = self.documents[did].sentences[0]
        else:
            sentence = self.documents[did].find_sentence_containing(start, end)
            if not sentence:
                print("could not find this sentence!", start, end)
        tokens = sentence.find_tokens_between(start, end)
        if not tokens:
            print("could not find tokens!", start, end, sentence.sid, ':'.join(res))
            sys.exit()
        entity = sentence.entities.find_entity(start - sentence.offset, end - sentence.offset)
        return tokens, sentence, entity

    def get_all_entities(self, source):
        entities = []
        for d in self.documents:
            for s in self.documents[d].sentences:
                for e in s.entities.elist[source]:
                    entities.append(e)
        return entities

    def clear_annotations(self, entitytype="all"):
        logging.info("Cleaning previous annotations...")
        for pmid in self.documents:
            for s in self.documents[pmid].sentences:
                if "goldstandard" in s.entities.elist:
                    del s.entities.elist["goldstandard"]
                if entitytype != "all" and "goldstandard_" + entitytype in s.entities.elist:
                    del s.entities.elist["goldstandard_" + entitytype]
                for t in s.tokens:
                    if "goldstandard" in t.tags:
                        del t.tags["goldstandard"]
                        del t.tags["goldstandard_subtype"]
                    if entitytype != "all" and "goldstandard_" + entitytype in t.tags:
                        del t.tags["goldstandard_" + entitytype]

    def get_invalid_sentences(self):
        pass

    def evaluate_normalization(self):
        scores = []
        for did in self.documents:
            for s in self.documents[did].sentences:
                if "goldstandard" in s.entities.elist:
                    for e in s.entities.elist.get("goldstandard"):
                        scores.append(e.normalized_score)
        print("score average: {}".format(sum(scores)*1.0/len(scores)))
        scores.sort()
        print(scores[0], scores[-1])

    def get_sentences(self, hassource=None):
        for did in self.documents:
            for sentence in self.documents[did].sentences:
                if hassource and 'goldstandard' in sentence.entities.elist:
                    yield sentence
                elif hassource is None:
                    yield sentence

