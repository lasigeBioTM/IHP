import codecs
import time
import logging
import glob
import os

from text.corpus import Corpus
#from text.document import Document
from text.hpo_document import HPODocument
from other.dictionary import stopwords, removewords, go_words, definingwords

annotation_gazette = open("data/annotation_gazette.txt")
ann_gaz = [x.strip() for x in annotation_gazette]

class SuiteCorpus(Corpus):
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
				print "could not find this sentence!", start, end
		tokens = sentence.find_tokens_between(start, end)
		if not tokens:
			print "could not find tokens!", start, end, sentence.sid, ':'.join(res)
			sys.exit()
		entity = sentence.entities.find_entity(start - sentence.offset, end - sentence.offset)
		return tokens, sentence, entity

	def get_offsets(self, esource, ths, rules):
		"""
		Retrieve the offsets of entities found with the models in source to evaluate
		:param esources:
		:return: List of tuple : (did, start, end, text)
		"""
		i = 0
		offsets = {} # {did1: [(0,5), (10,14)], did2: []...}
		for did in self.documents:
			i += 1
			print self.documents[did].sentences, i
			offsets[did] = self.documents[did].get_offsets(esource, ths, rules)
		offsets_list = []
		for did in offsets:
			for o in offsets[did]:
				offsets_list.append((did, o[0], o[1], o[2]))
				#print did, o[0], o[1], o[2]


		offsets_list = list(set(offsets_list))
		off = [str(x[3].encode("utf-8")) for x in offsets_list] #List of terms to be passed for second validation
		#Twice Validated (get_offsets validated at the same time as getting results if rules are activated)
		if "twice_validated" in rules:
			#corenlp_client = StanfordCoreNLP('http://localhost:9000')
			for did in self.documents:	
				offsets[did] = self.documents[did].get_offsets(esource, ths, rules, off)
			for did in offsets:
				for o in offsets[did]:
					offsets_list.append((did, o[0], o[1], o[2]))
			offsets_list = list(set(offsets_list))

			#Validation step -> Removal of wrong terms
			to_remove = self.validate_corpus(offsets_list, rules)
			for off in list(set(to_remove)):
				offsets_list.remove(off)

		return offsets_list


	def validate_corpus(self, offsets_list, rules):
		to_remove = []
		for offset in offsets_list:
			if str(offset[3].encode("utf-8")).lower() in stopwords:
				to_remove.append(offset)
			for word in removewords:
				for w in go_words:
					if word.lower() in str(offset[3].encode("utf-8").lower()) and w.lower() not in str(offset[3].encode("utf-8").lower()):
						try:
							to_remove.append(offset)
						except ValueError:
							pass
			if "small_len" in rules:
				if len(str(offset[3].encode("utf-8")).lower()) < 3:
					to_remove.append(offset)
			if "quotes" in rules:
				try:
					if '"' in str(offset[3].encode("utf-8")).lower():
						if str(offset[3].encode("utf-8")).lower().count('"') == 1:
							to_remove.append(offset)
				except UnicodeDecodeError:
					pass
			if "defwords" in rules:
				flag = False
				for word in definingwords:
					if word in str(offset[3].encode("utf-8")).lower():
						flag = True
				if flag:
					if len(str(offset[3].encode("utf-8")).split(" ")) < 3:
						to_remove.append(offset)

			if "digits" in rules:
				try:
					for x in str(offset[3].encode("utf-8")).split(" "):
						flags = [False for a in range(len(x))]
						for i in range(len(x)):
							flags[i] = x[i] in "0123456789," #",.;:!?." in there because might be after number
						digflag = True
						for f in flags:
							if f == False:
								digflag = False
						if digflag == True:
							to_remove.append(offset)
				except UnicodeDecodeError:
					pass
			if "gowords" in rules: #Removes terms that have 2 of the go words because it doesn't really happen.
				i = 0
				for word in go_words:
					if word in str(offset[3].encode("utf-8")):
						i += 1
				if i > 1:
					to_remove.append(offset)

			# if "check_nouns" in rules:
			# 	if len(str(offset[3].encode("utf-8")).split(" ")) > 2 and str(offset[3].encode("utf-8")).lower().strip() not in ann_gaz:
			# 		corenlpres = corenlp_client.annotate(str(offset[3].encode("utf-8")), properties={
			# 						'annotators': 'tokenize,pos',
			# 						'outputFormat': 'json',
			# 					})
			# 		toks = corenlpres['sentences'][0]['tokens']
			# 		noun_flag = False
			# 		postags = [ str(x['pos']) for x in toks]
			# 		if "NN" in postags or "NNS" in postags: #or "VBG" in postags:
			# 			noun_flag = True
			# 		if noun_flag == False:
			# 			print offset
			# 			to_remove.append(offset)

			if "lastword" in rules:
				from pycorenlp import StanfordCoreNLP
				corenlp_client = StanfordCoreNLP('http://localhost:9000')
				exlude_last = ['-LRB-', 'TO', 'IN', 'PRP', '.', '.', 'CC', 'DT']
				corenlpres = corenlp_client.annotate(str(offset[3].encode("utf-8")), properties={
									'annotators': 'tokenize,pos',
									'outputFormat': 'json',
								})
				toks = corenlpres['sentences'][0]['tokens']
				postags = [ str(x['pos']) for x in toks]
				#words = postags = [ str(x['word']) for x in toks]
				if len(postags) > 1:
					if postags[-1] in exlude_last:
						print "Last word removed", offset
						to_remove.append(offset)
					if postags[-1] == "JJ" and postags[-2] in exlude_last:
						print "Last word ADJ removed", offset
						to_remove.append(offset)

		return to_remove


#Bracket (left) -  '-LRB-'
#To - 'TO'
#In, Within -  'IN'
#Prepositions -  'PRP'
#. - '.'
#, - ','


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