from text.document import *
from text.sentence import Sentence
#from text.hpo_sentence import HPOSentence

class HPODocument(Document):
    """A document is constituted by one or more sentences. It should have an ID and
    title. s0, the first sentence, is always the title sentence."""

    def __init__(self, text, process=False, doctype="biomedical", ssplit=False, **kwargs):
        super(HPODocument, self).__init__(text, process=False, doctype="biomedical", ssplit=False, **kwargs)
        self.text = text
        self.title = kwargs.get("title")
        self.sentences = kwargs.get("sentences", [])
        self.did = kwargs.get("did", "d0")
        if ssplit:
            self.sentence_tokenize(doctype)
        if process:
            self.process_document(doctype)

    # def sentence_tokenize(self, doctype):
    #     """
    #     Split the document text into sentences, add to self.sentences list
    #     :param doctype: Can be used in the future to choose different methods
    #     """
    #     # first sentence should be the title if it exists
    #     a = 1
    #     if self.title:
    #         a = 0
    #         sid = self.did + ".s0"
    #         self.sentences.append(HPOSentence(self.title, sid=sid, did=self.did))
    #     # inputtext = clean_whitespace(self.text)
    #     inputtext = self.text
    #     with codecs.open("/tmp/geniainput.txt", 'w', 'utf-8') as geniainput:
    #         geniainput.write(inputtext)
    #     current_dir = os.getcwd()
    #     os.chdir(geniass_path)
    #     geniaargs = ["./geniass", "/tmp/geniainput.txt", "/tmp/geniaoutput.txt"]
    #     Popen(geniaargs, stdout=PIPE, stderr=PIPE).communicate()
    #     os.chdir(current_dir)
    #     offset = 0
    #     with codecs.open("/tmp/geniaoutput.txt", 'r', "utf-8") as geniaoutput:
    #         for l in geniaoutput:
    #             stext = l.strip()
    #             sid = self.did + ".s" + str(len(self.sentences) + a) ######### CAN THIS HAPPEN?
    #             self.sentences.append(HPOSentence(stext, offset=offset, sid=sid, did=self.did))
    #             offset += len(stext)
    #             offset = self.get_space_between_sentences(offset)

    def tag_hpo_entity(self, start, end, **kwargs):
        """
        Create an HPO entity relative to this document. It iterates the sentences
        and checks for entities. If it finds, it breaks. Goes to the next one.
        :param start: Start index of entity
        :param end: End index of entity
        :param kwargs: Extra stuff like the text
        :return:
        """
        doct = kwargs.get("doct")
        if doct == "T": # If it's in the title, we already know the sentence (it's the first)
            self.sentences[0].tag_hpo_entity(start, end, **kwargs)
        else: # we have to find the sentence
            found = False
            totalchars = 0
            for s in self.sentences:
                if totalchars <= start and totalchars + len(s.text) >= end:  # entity is in this sentence
                    s.tag_entity(start-totalchars, end-totalchars, etype="hpo", **kwargs) #totalchars=totalchars,
                    # print "found entity on sentence %s" % s.sid
                    found = True
                    break

                totalchars += len(s.text)
                totalchars = self.get_space_between_sentences(totalchars)
            if not found:
                print "could not find sentence for %s:%s on %s!" % (start, end, self.did)
                # sys.exit()

    def write_hpo_results(self, source, outfile, ths={"hpo":0.0}, rules=[]):
        lines = []
        totalentities = 0
        for s in self.sentences:
            # print "processing", s.sid, "with", len(s.entities.elist[source]), "entities"
            if s.entities:
                res = s.entities.write_hpo_results(source, outfile, ths, rules, totalentities+1)
                lines += res[0]
                totalentities = res[1]
        return lines
