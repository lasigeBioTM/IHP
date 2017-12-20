
import codecs
import time
import logging
import glob
import os

from text.corpus import Corpus
#from text.document import Document
from text.hpo_document import HPODocument
from other.dictionary import stopwords, removewords, go_words, definingwords, good_nouns


same_stop_words = [""]
describing = ["recurrent", "male", "female", "postnatal", "progressive", "isolated", "postpubertal", "severe", "distal", "conductive", "mixed", "congenital", "bilateral", "unilateral", "chronic", "episodic", "mild", "borderline-mild", "global", "generalized", "partial", "acute", "proximal", "profound", "complete", "moderate", "diffuse", "nonprogressive", "extreme", "general"]

annotation_gazette = open("data/annotation_gazette.txt")
gazette = open("data/gazette.txt")
ann_gaz = [x.strip() for x in annotation_gazette]
gazz = [x.strip() for x in gazette]

class HPOCorpus(Corpus):
	"""Human Genotype Ontology corpus - Example (http://bio-lark.org/hpo_res.html)"""

	def __init__(self, corpusdir, **kwargs): #corpusdir in config.py
		super(HPOCorpus, self).__init__(corpusdir, **kwargs)

	def load_corpus(self, corenlpserver, process=True):
		""" Loads HPO corpus with the corpusdir element
			corenlpserver -> StanfordCoreNLP from https://bitbucket.org/torotoki/corenlp-python
			process = True/False. If true it will use process_document in the document
			created.
		"""
		#Each file in the corpus dir has 1 line. Need to go through all files and take text. Each filename is ID. No title
		total_lines = sum(1 for file in glob.glob(self.path + "/*")) #number of files containing 1 line
		number_of_lines = 1
		time_per_abstract = []
		#print self.path
		for file in glob.glob(self.path + "/*"):
			#print file
			line = codecs.open(file, "r", "utf-8").read()
			t = time.time()
			did = file.split("/")[-1]

			doctext = line.strip().replace("<", "(").replace(">", ")")
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


	#Each HPO Annotations:
		#[27::42]	HP_0000110 | renal dysplasia

	def load_annotations(self, ann_dir, etype="hpo", ptype="all"): #annotation directory
		logging.info("Cleaning previous annotations...")
		for pmid in self.documents:
			sentences = self.documents[pmid].sentences
			for sentence in sentences:
				if "goldstandard" in sentence.entities.elist:
					del sentence.entities.elist["goldstandard"]

		logging.info("Loading annotations file...")
		for file in glob.glob(ann_dir + "/*"):
			pmid = file.split("/")[-1]

			annotations = codecs.open(file, "r", "utf-8")
			for line in annotations:
				elements = line.strip().split("\t")
				
				off = elements[0].split("::")
				start = off[0][1:]
				end = off[1][:-1]

				oth = elements[1].split(" | ")
				id = oth[0]
				text = oth[1]
				doct = "A" ##################	has to depend if title or not #######

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
		offsets = {} # {did1: [(0,5), (10,14)], did2: []...}
		for did in self.documents:
			offsets[did] = self.documents[did].get_offsets(esource, ths, rules)
		offsets_list = []
		for did in offsets:
			for o in offsets[did]:
				offsets_list.append((did, o[0], o[1], o[2]))
				#print did, o[0], o[1], o[2]


		offsets_list = list(set(offsets_list))
		off = [str(x[3].encode("utf-8")) for x in offsets_list] #List of terms to be passed for second validation
		#print "malformations of the kidney" in off
		#Twice Validated (get_offsets validated at the same time as getting results if rules are activated)
		if "twice_validated" in rules:
			for did in self.documents:	
				offsets[did] = self.documents[did].get_offsets(esource, ths, rules, off)
			for did in offsets:
				for o in offsets[did]:
					offsets_list.append((did, o[0], o[1], o[2]))
			offsets_list = list(set(offsets_list))

			#Validation step -> Removal of wrong terms
			to_remove = self.validate_corpus(offsets_list, rules)
			for offa in list(set(to_remove)):
				offsets_list.remove(offa)

			#Validation step -> Removal of wrong terms
			to_remove = self.validate_corpus(offsets_list, rules)
			for off in list(set(to_remove)):
				offsets_list.remove(off)

		#off = [str(x[3].encode("utf-8")) for x in offsets_list] #List of terms to be passed for second validation

		#print "malformations of the kidney" in off
		return offsets_list


	def validate_corpus(self, offsets_list, rules):
		#offa = [str(x[3].encode("utf-8")) for x in to_remove] #List of terms to be passed for second validation
		#print "neurofibromatosis" in offa
		to_remove = []
		if "gen_rules" in rules and "removal" in rules:
			for offset in offsets_list:
				if "stopwords" in rules:
					if str(offset[3].encode("utf-8")).lower() in stopwords:
						to_remove.append(offset)
					for word in removewords:
						for w in go_words:
							if word.lower() in str(offset[3].encode("utf-8").lower()) and w.lower() not in str(offset[3].encode("utf-8").lower()):
								try:
									to_remove.append(offset)
								except ValueError:
									pass

				if "gen_errors" in rules:
					#Remove entities smaller than 3 characters
					if len(str(offset[3].encode("utf-8")).lower()) < 3:
						print "small length removed", offset
						to_remove.append(offset)

					#Remove words that only have one double-quote or parenthesis
					try:
						if '"' in str(offset[3].encode("utf-8")).lower():
							if str(offset[3].encode("utf-8")).lower().count('"') == 1:
								print "qt removed", offset
								to_remove.append(offset)
						if ')' in str(offset[3].encode("utf-8")).lower() or '(' in str(offset[3].encode("utf-8")).lower():
							cou = str(offset[3].encode("utf-8")).lower().count(')') + str(offset[3].encode("utf-8")).lower().count('(')
							if cou != 2:
								print "qt removed", offset
								to_remove.append(offset)							
					except UnicodeDecodeError:
						pass
					
					#Remove entities that have less than 3 words and that include "type" or "group"
					flag = False
					for word in definingwords:
						if word in str(offset[3].encode("utf-8")).lower():
							flag = True
					if flag:
						if len(str(offset[3].encode("utf-8")).split(" ")) < 3:
							print "defword removed", offset
							to_remove.append(offset)

					#Remove entities that contain digits
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
								print "digits removed", offset
								to_remove.append(offset)
					except UnicodeDecodeError:
						pass


					#Removes Entities that have 2 of the go words because it doesn't really happen.
					go_count = 0
					for word in go_words:
						if (word == str(offset[3].encode("utf-8")).lower().split(" ")[0] or
						word == str(offset[3].encode("utf-8")).lower().split(" ")[-1]):
							go_count += 1
					if go_count > 1:
						print "gowords removed", offset
						to_remove.append(offset)

					#Removes Entities that contain 2 words and have a comma
						# if str(offset[3].encode("utf-8")).count(",") == 1 and str(offset[3].encode("utf-8")).count("and") == 1:
						# 	print "and_comma removed", offset
						# 	to_remove.append(offset)
					if len(str(offset[3].encode("utf-8")).split(" ")) <= 2 and "," in str(offset[3].encode("utf-8")):
						print offset
						to_remove.append(offset)

				if "negcon" in rules:
					#Removes entities that are smaller than 3 and coontain a positive noun. Since we're dealing with disorders
					#It's necessary to have at least 3 words to give a negative connotaction to positive nouns
					for noun in good_nouns:
						if noun in str(offset[3].encode("utf-8")) and len(str(offset[3].encode("utf-8")).split(" ")) < 3:
							print "good_nouns word removed", offset
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

				#if "small_ent" in rules: #Remove smaller entities
					#smaller_entities = set([])
				#	for b in [str(x[3].encode("utf-8")) for x in offsets_list]:
				#		if (str(offset[3].encode("utf-8")) in b and str(offset[3].encode("utf-8")) != b and
				#		    str(offset[3].encode("utf-8")).lower() not in ann_gaz+gazz and 
				#		    len(str(offset[3].encode("utf-8")).split(" ")) != len(b.split(" "))):
								#print a[2].lower()
				#				print "{}: small entity: {} / big entity: {}".format(str(offset[0].encode("utf-8")), str(offset[3].encode("utf-8")), b)

	#							logging.info("{}: small entity: {} / big entity: {}".format(str(offset[0].encode("utf-8")), str(offset[3].encode("utf-8")), b))
				#				to_remove.append(offset)
								#smaller_entities.add(a)
					#for x in smaller_entities:
					#	sentence_terms.remove(x)

				if "lastwords" in rules:
					from pycorenlp import StanfordCoreNLP
					corenlp_client = StanfordCoreNLP('http://localhost:9000')
					exlude_last = ['-LRB-', 'TO', 'IN', 'PRP', 'PRP$', '.', ',', 'CC', 'DT']
					exception_list = ["apnea", "antihelix"]
					corenlpres = corenlp_client.annotate(str(offset[3].encode("utf-8")), properties={
										'annotators': 'tokenize,pos',
										'outputFormat': 'json',
									})
					toks = corenlpres['sentences'][0]['tokens']
					postags = [ str(x['pos']) for x in toks]
					words = [ str(x['word'].encode("utf-8")) for x in toks]
					if len(postags) > 1:
						if postags[-1] in exlude_last and words[-1] not in exception_list:
							print "Last word removed", offset
							to_remove.append(offset)
						if postags[-1] in ["JJ", "JJR", "JJS"] and postags[-2] in exlude_last and words[-1] not in describing:
							print "Last word ADJ removed", offset
							to_remove.append(offset)
						if postags[-1] == "RB" and postags[-2] in exlude_last:
							print "Last word removed", offset
							to_remove.append(offset)

						#First words
						if postags[0] == "DT":
							print "Last word removed", offset
							to_remove.append(offset)
						if postags[-1] == ",":
							print "Last word removed", offset
							to_remove.append(offset)

						#Last character
						if str(offset[3].encode("utf-8"))[-1] == "-":
							print "Last word removed", offset
							to_remove.append(offset)

						#Last word
						lwords = ["has", "have", "is", "had"]	
						if str(offset[3].encode("utf-8")).split(" ")[-1] in lwords:
							print "Last word removed", offset
							to_remove.append(offset)					

						#Create one that removes adjectives (length 1 entity) that are not in gazette.
					if len(postags) == 1:
						if postags[0] in ["JJ", "JJR", "JJS"]:
							if words[0] not in gazz+ann_gaz:
								to_remove.append(offset)
								print "Adjective removed: {}".format(words[0])


	 		to_remove = set(list(to_remove))
		return to_remove

def hpo_get_gold_ann_set(goldpath): #goldann="corpora/hpo/test_ann"
	"""
	Load the HPO annotations to a set
	:param goldann: Path to HPO annotation file folder (several files)
	:return: Set of gold standard annotations
	"""
	# TODO: copy to chemdner:corpus
	goldlist = []
	for file in glob.glob(goldpath + "/*"):
		pmid = file.split("/")[-1]
		annotations = codecs.open(file, "r", "utf-8")
		for line in annotations:
			elements = line.strip().split("\t")
				
			off = elements[0].split("::")
			start = off[0][1:]
			end = off[1][:-1]

			oth = elements[1].split(" | ")
			id = oth[0]
			text = oth[1]
			doct = "A"
			#pmid, T/A, start, end
		   # goldlist.append((pmid, doct + ":" + start + ":" + end, '1'))
			goldlist.append((pmid, int(start), int(end), text))

	#print goldlist[0:2]
	goldset = set(goldlist)
	#print goldset
	return goldset, None

