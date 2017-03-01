import sys
import os
import multir_pb
from classification.rext.kernelmodels import ReModel

class MultiR(ReModel):
    def __init__(self, corpus, relationtype, modelname="multi_classifier.ser"):
        super(MultiR, self).__init__()
        self.modelname = modelname
        self.pairs = {}
        self.corenlp_client = None
        self.relationtype = relationtype
        self.corpus = corpus
        self.generate_data(corpus,modelname,relationtype)

    def generate_data(self, corpus, modelname, pairtypes):
        # TODO: refactor this part to corpus class
        if os.path.isfile(self.temp_dir + modelname + ".pb"):
            print "removed old data"
            os.remove(self.temp_dir + modelname + ".pb")
        trainlines = []
        # get all entities of this document
        # doc_entities = []
        pcount = 0
        truepcount = 0
        ns = 0
        nentities = 0
        doc = multir_pb.Document()
        doc.filename = modelname
        for sentence in corpus.get_sentences("goldstandard"):
            sent = doc.sentences.add()
            for token in sentence.tokens:
                tok = sent.tokens.add()
                tok.word = token.text
                tok.tag = token.pos
                token.ner = token.tag
            for entity in sentence.entities.elist.get("goldstandard"):
                ent = sent.mentions.add()
                ent.id = nentities
                ent.from_ = entity.start
                ent.to = entity.end
                ent.label = entity.text
                nentities += 1
        # Write the new address book back to disk.
        f = open("train.pb.gz", "wb")
        f.write(doc.SerializeToString())
        f.close()



