import codecs
import time
import logging
import glob
import os

from reader.hpo_corpus import HPOCorpus
#from text.document import Document
from text.hpo_document import HPODocument
from other.dictionary import stopwords, removewords, go_words, definingwords, good_nouns

same_stop_words = [""]
describing = ["recurrent", "male", "female", "postnatal", "progressive", "isolated", "postpubertal", "severe", "distal", "conductive", "mixed", "congenital", "bilateral", "unilateral", "chronic", "episodic", "mild", "borderline-mild", "global", "generalized", "partial", "acute", "proximal", "profound", "complete", "moderate", "diffuse", "nonprogressive", "extreme", "general"]
annotation_gazette = open("data/annotation_gazette.txt", encoding='utf-8')
gazette = open("data/gazette.txt", encoding='utf-8')
ann_gaz = [x.strip() for x in annotation_gazette]
gazz = [x.strip() for x in gazette]
class SuiteCorpus(HPOCorpus):
	"""Human Genotype Ontology corpus - Example (http://bio-lark.org/hpo_res.html)"""

	def __init__(self, corpusdir, **kwargs): #corpusdir in config.py
		super(SuiteCorpus, self).__init__(corpusdir, **kwargs)

	def load_corpus(self, corenlpserver, process=True):
		""" Loads HPO corpus with the corpusdir element
			corenlpserver -> StanfordCoreNLP from https://bitbucket.org/torotoki/corenlp-python
			process = True/False. If true it will use process_document in the document
			created.
		"""
		#Each file in the corpus dir has 1 line. Need to go through all files and take text. Each filename is ID. No title
		total_lines = 0
		for file in glob.glob(self.path + "/*"):
			test_file = codecs.open(file, "r", "utf-8").read()
			tests = test_file.split("\n\n")	
			for test in tests:
				if len(test) > 2:
					lines = test.split("\n")
					for line in lines:
						if line.startswith(" - "):
							total_lines += 1

		number_of_lines = 1
		time_per_abstract = []
		#print self.path
		for file in glob.glob(self.path + "/*"):
			
			t = time.time()
			test_file = codecs.open(file, "r", "utf-8").read()
			tests = test_file.split("\n\n")	
			for test in tests:
				if len(test) > 2:
					
					title = test[test.find("#"):test.find("\n")][2:].strip("\n")
					i = 0
					lines = test.split("\n")
					for line in lines:
						if line.startswith(" - "):
							i += 1
							did = file.split("/")[-1] + "." + title + "." + str(i) 
							term = line.replace(" - ", "").strip("\n")
							term2 = term[term.find("=")+1:].strip()
							#print term2, i

							doctext = term2.strip().replace("<", "(").replace(">", ")")
							newdoc = HPODocument(doctext, process=False,
							   							  did=did)

							logging.info("processing " + newdoc.did + 
										 ": " + str(number_of_lines) + 
										 "/" + str(total_lines))
							newdoc.sentence_tokenize("biomedical")
							if process:
								newdoc.process_document(corenlpserver, "biomedical")
							self.documents[newdoc.did] = newdoc
							number_of_lines += 1
							abstract_time = time.time() - t
							time_per_abstract.append(abstract_time)
							logging.info("%s sentences, %ss processing time" %
										 (len(newdoc.sentences), abstract_time))

							abstract_average = sum(time_per_abstract)*1.0/len(time_per_abstract)
							logging.info("average time per abstract: %ss" % abstract_average)


	def load_annotations(self, ann_dir, etype="hpo", ptype="all"): #annotation directory
		logging.info("Cleaning previous annotations...")
		for pmid in self.documents:
			sentences = self.documents[pmid].sentences
			for sentence in sentences:
				if "goldstandard" in sentence.entities.elist:
					del sentence.entities.elist["goldstandard"]

		logging.info("Loading annotations file...")
		for file in glob.glob(self.path + "/*"):
			
			t = time.time()
			test_file = codecs.open(file, "r", "utf-8").read()
			tests = test_file.split("\n\n")	
			for test in tests:
				if len(test) > 2:
					title = test[test.find("#"):test.find("\n")][2:].strip("\n")
					i = 0
					lines = test.split("\n")
					for line in lines:
						if line.startswith(" - "):
							i += 1
							pmid = file.split("/")[-1] + "." + title + "." + str(i) 
							term = line.replace(" - ", "").strip("\n")
							text = term[term.find("=")+1:].strip()
							#print text, i
							start = 0
							end = len(text)
							doct = "A"
							#print text, str(start), str(end)
							if pmid in self.documents:
								self.documents[pmid].tag_hpo_entity(int(start), int(end), text=text, doct=doct)
							else:
								logging.info("%s not found!" % pmid)


	def write_hpo_results(self, source, outfile, ths={"chebi":0.0}, rules=[]):
		"""
		Produce results to be evaluated with the BioCreative hpo evaluation script
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
			doclines = self.documents[d].write_hpo_results(source, outfile, ths, rules)
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

	def find_hpo_result(self, res):
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


def tsuite_get_gold_ann_set(goldpath): #goldann="corpora/hpo/test_ann"
	"""
	Load the HPO annotations to a set
	:param goldann: Path to HPO annotation file folder (several files)
	:return: Set of gold standard annotations
	"""
	# TODO: copy to chemdner:corpus
	goldlist = []
	for file in glob.glob(goldpath + "/*"):
		
		t = time.time()
		test_file = codecs.open(file, "r", "utf-8").read()
		tests = test_file.split("\n\n")	
		for test in tests:
			if len(test) > 2:
				i = 0
				title = test[test.find("#"):test.find("\n")][2:].strip("\n")
				
				lines = test.split("\n")
				for line in lines:
					if line.startswith(" - "):
						term = line.replace(" - ", "").strip("\n")
						text = term[term.find("=")+1:].strip()
						start = 0
						end = len(text)
						doct = "A"
						i += 1
						pmid = file.split("/")[-1] + "." + title + "." + str(i)
						goldlist.append((pmid, int(start), int(end), text))

	#print goldlist[0:2]
	goldset = set(goldlist)
	#print goldset
	return goldset, None
