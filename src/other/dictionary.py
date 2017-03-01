import urllib2
from bs4 import BeautifulSoup
import nltk
import logging

from pycorenlp import StanfordCoreNLP
corenlp_client = StanfordCoreNLP('http://localhost:9000')

stopwords = ["all", "high", "left", "lower", "low", "right", "central", "severe", "mild", "large", "onset",
			 "bilateral", "multiple", "profound", "proximal", "distal", "foot", "long", "papules", "middle",
			 "position", "congenital", "inheritance", "small", "genitalia", "brachypodism",
			 "localized", "ear", "prominent", "peripheral", "mosaicism", "fibroma", "arrhythmia", 
			 "pterygium", "hamartoma", "nevus", "neuroma", "severity", "approximately", #schwanomma used several times in some cases.
			  "attenuation", "age of onset", "nevi", "BDC", "unilateral", "neurinoma", "learning",
			 "macrocephaly", "progressive", "soft", "heterozygosity", "speech", "audiometry", "generalized",
			 "stereotyped", "fine", "diffuse", "carbamazepine", "oxcarbazepine", "interview",
			 "frequent", "infections", "wide", "fluorescence", "hybridization", "tumorigenesis", "anemia",
			 "helix", "audiometry", "papilloma", "microduplications", "mitochondrial", "IHH", "irregular",
			 "localised", "mononeuropathy", "naevus", "proliferation", "deletion", "thin", "epicanthal", "kidneys",
			 "broad", "flattened", "echocardiography", "numbness", "asymmetric", "Deletion", "nonspecialty",
			 "microsatellite", "persistent", "metacarpal", "schwannomin", "hybridisation", "secretion", "coronoid", 
			 "episodes", "vocalization", "configuration", "acute", "membrane", "activities"]
			  #nonprogressive should also be added even if it reduces score...

removewords = [';', '!', '?', 'author', ' function', ' ability', ' growth', ' evaluations', ' laboratory',
			   ' recognition', 'mice', 'trafficking', ' cloning', ' ultrasonography', ' pyelography',
			   ' ultrasound', ' treatment', ' activator', ' class', ' assay', ' domain', ' strategy', ' substrate',
			   'examination', ' translocation', ' fate', ' marker', ' effect', ' follicles', ' pyelograms', ' microscopy',
			   ' hybrids', ' investigations', 'techniques', ' capsule', ' cluster', ' correlation', 
			   ' interface', 'stochastic', 'contact', 'parameters', 'error', 'triplications', 'alleles', 'duplications',
			   'videotaping', 'digitized', 'useful', ' outcome', 'audiogram', 'immunotherapy',
			   ' synthetic', 'understand', 'mechanism', 'irradiation', 'detection', 'hybridisation', 'RRB', 'regulator',
			   'maintenance', 'microsatellite', 'magnetic', 'testing', 'patency', 'classic', 'neuroanatomy', 'inducible',
			   'tandem', 'likely', 'various', 'neurogenetic', 'categorization', 'habilitative', 'vitro',
			   'aneuploid', 'mitoses', 'radiotherapy', 'exenteration', 'association', 'environmental', 'contribution',
			   'allelic', 'pedigree', 'apparent', 'ancestors', 'history', 'estimates', 'explanation', 'extended',
			   'expansion', 'extracellular', 'prevalent', 'splicing', 'identical', 'distinctly', 'experiments',
			   'lethal damage', 'survey', 'report', 'hypothesis', 'counseling', 'biparental', 'biopsy',
			   'database', 'inhibit ', 'isodisomy', 'exploratory', 'uniparental', 'disomy', 'while', 'identifiable', 
			   'known', 'causes', 'results', 'review', 'subsequent', 'nonpaternity', 'consistency', 'distinct ',
			   'radiological', 'comparing', 'sample', 'breakpoint', 'hybridization', 'domains', 'assessment',
			   'clinicians', 'expert', 'microarray', 'pathways', 'possible', 'international', 'meeting', 'routine ',
			   'MluI', 'divergence', 'immunohistochemistry', ' sib ', 'exon', 'signaling', 'colonoscopy', 'visible', 'acrosome',
			   'brother', 'father', 'methylation', 'mendelian', 'polyphasic', 'thoracotomy', 'caregiver', 'retrospective',
			   '15q', '13q', '9q31', 'neurofilament', 'electrophoresis', 'insult', 'obvious', 'hairbulb', 'architecture', 'copy',
			   'complicated', 'resection', 'designated', 'mapping', ' genes', 'flame', 'girl', 'boy', 'highlight', 'years',
			   'analysis', 'daughter', 'mother', 'contacts', 'centromeric', 'anesthesia', 'postponement', 'operation', 'heterozygosity',
			   'multilobular', 'diverse', 'nonsense', 'lifetime', 'suppressor', 'fibroblasts', ' this ', 'imprinted',
			   'critical', 'maternally', 'offspring', 'risks', 'postmitotic', 'conventional', 'attachments', 'widespread', 
			   'majority', 'acromesomelic', 'surgical', 'procedures', 'which', ' only', 'points', 'cytogenetic', 'fusing',
			   'proposita', 'suggestive', 'previous', 'analyses', 'mushroom', 'clumsy', 'spares', 'hedgehog', 'elements',
			   'twin', 'physiologic', 'bushman', 'maori', 'equivalents', 'transcriptional', 'coactivator', 'carcinogenesis',
			   'postmitotic', 'pseudodiploids', 'bifrontally', 'troponin', 'quantitative', 'phenobarbital', 'psychological',
			   'candidate', 'ube3a', 'suppression', ] #clash with regulatory (once in gazette)
			   #author, authors #"function" works in conjunction with go_words to remove thWe 
																	  #common use of the word function
		 
pre_entity_terms = ["mild to severe", "mild to moderate", "mild to profound",
					"moderate to severe", "moderate to profound", "severe to profound"]

go_words = ["abnormalities", "anomalies","involvement", "abnormality", #"anomaly",
			"disturbances", "immaturity", "defects", "absence", "malformations", "duplication",
			"problems", "lesions", "syndromes", 'symptoms', 'calcification', 'loss', 'delay'] #, "manifestations"] #Need to fix for ocular manifestations for example-
pos_go_words = ["of the", "in the", "of"]
suffixes = ["ous", "al", "aly", "dly", "tly"]

definingwords = ["group", "type"]
non_words = ["in", "to", "of", "the"]


#simple_terms = [] #LDDB List containing terms with only 2 tokens
#a = open("data/LDDB2HPO_terms.txt")
#for line in a:
#	t = line.strip()
#	simple_terms.append(t)


class Dictionary():
	def __init__(self):
		self.list_of_terms = open("data/gazette.txt").readlines()
		self.term_dic = {}

		annotation_gazette = open("data/annotation_gazette.txt").readlines()
		# a = "low-molecular-weight proteinuria"
		# print nltk.word_tokenize(a)
		# corenlpres = corenlp_client.annotate(a.encode("utf8"), properties={
  #               # 'annotators': 'tokenize,ssplit,pos,depparse,parse',
  #               'annotators': 'tokenize,pos,parse',
  #               'outputFormat': 'json',
  #           })
		# toks = corenlpres['sentences'][0]['tokens']
		# y = [ str(x['word']) for x in toks]
		# print y

		# x = "assasfa"
		# print x[-2]

		for x in self.list_of_terms:#+ annotation_gazette:
			term = x.strip().lower()
			tokenized = nltk.word_tokenize(term)
			#for token in tokenized:
			#	self.term_dic[token] = []
			#if "molecular" in tokenized[0]:
				#print tokenized[0]
			self.term_dic[tokenized[0]] = []
			self.term_dic[tokenized[-1]] = []

		for x in self.list_of_terms:# + annotation_gazette:
			term = x.strip().lower()
			tokenized = nltk.word_tokenize(term)
			self.term_dic[tokenized[0]].append(term)
			self.term_dic[tokenized[-1]].append(term)
			#for token in tokenized:
			#	self.term_dic[token].append(term)
		#for x in self.term_dic:
		#	self.term_dic[x] = list(set(self.term_dic[x]))


	def create_dictionary(self):
		for term in self.list_of_terms:
			self.dictionary[term] = [term.strip()]

		for term in self.list_of_terms:
			synonym_list = self.find_synonyms(term.strip())
			for syn_term in synonym_list:
				self.dictionary[term].append(syn_term)

	def search_terms(self, sentence):
		sentence_terms = []
		for term in self.dictionary:
			for syn in self.dictionary[term]:
				if term in sentence:
					start = sentence.index(term)
					end = start + len(term)
					sentence_terms.append((term, start, end))
		return sentence_terms

	def easy_search_terms(self, sentence, esource, ths, rules, off_list=None):
		""" Search new entities using a dictionary. Checks if the
			entities are already checked
			Sentence is an IBEnt Sentence Object"""
		reduced_term_list = []
		sentence_terms = []
		sentence_terms_text = []
		entity_tokens = []


		#Create a gazette with words that start with any word of the sentence
		#Why was this after next section?
		for token in sentence.tokens:
			ptoken = str(token.text.encode("utf-8")).lower()
			if ptoken in self.term_dic:
				for term in self.term_dic[ptoken]:
					#print term
					reduced_term_list.append(term)

		if off_list != None:
			if "twice_validated" in rules:
				for term in off_list + reduced_term_list:
					tsplit = term.split(" ")
					if len(tsplit) == 2:
						t = tsplit[1] + " in the " + tsplit[0]
						t2 = tsplit[1] + " of the " + tsplit[0]
						#print sentence.sid, t
						if t in sentence.text.lower() or t2 in sentence.text.lower():
							#print t
							reduced_term_list.append(tsplit[1] + " in the " + tsplit[0])
							reduced_term_list.append(tsplit[1] + " of the " + tsplit[0])
					if "type" in term:
						i = term.find("type")
						x = term[:i].strip()
						y = term[i:].strip()
						reduced_term_list.append(x + " " + y)
						reduced_term_list.append(x + " " + y.replace("type", ""))
						reduced_term_list.append(y + " " + x)
						reduced_term_list.append(y + " " + x.replace("type", ""))



		#Iterate gazette and check if the sentence has any exact match
		for term in reduced_term_list:
			term_tokens = nltk.word_tokenize(term.lower())
			token_flag = False
			for token in sentence.tokens:
				if term_tokens[0].lower() == token.text.lower():
					token_flag = True
					start_index = sentence.tokens.index(token)
			if token_flag:					
				end_index = start_index + len(term_tokens)
				entity_tokens = []
				for token in sentence.tokens[start_index:end_index]:
					entity_tokens.append(token.text)
				#Check if gazette term is the same as sentence tokens for that term.								
				if term_tokens == [x.lower() for x in entity_tokens]:
					start = sentence.tokens[start_index:end_index][0].dstart
					end = sentence.tokens[start_index:end_index][-1].dend
					term = str(sentence.text[sentence.tokens[start_index:end_index][0].start:sentence.tokens[start_index:end_index][-1].end])					
					sentence_terms.append((start, end, term))
					sentence_terms_text.append(term)
					#print term
					

		### RULES FOR VALIDATION (maybe should be in functions)
		#Undidented next line. See if it changes reslts.
		for token in sentence.tokens: 	# sentence and adds the combination and next word
			if "posgowords" in rules: #Tries to find a combination of go_word and pos_go_word in
				pos_flag = False			# if next word is not a noun, looks for next word.
				tok = str(token.text.encode("utf-8").strip().lower())
				if tok in go_words:
					tlist = []
					for word in pos_go_words:
						term = tok + " " + word
						if term in str(sentence.text.encode("utf-8")):
							tlist.append(term)
					if len(tlist) > 0:
						term = max(tlist, key=len)
						l = len(term.split(" "))
						index_start = sentence.tokens.index(token)
						index_end = index_start + l + 1 #+1 for next word
						term = sentence.text[sentence.tokens[index_start:index_end][0].start:sentence.tokens[index_start:index_end][-1].end]
						if sentence.tokens[index_end-1].pos != "NN" and sentence.tokens[index_end-1].pos != "NNS": #
							index_end += 1
							term = sentence.text[sentence.tokens[index_start:index_end][0].start:sentence.tokens[index_start:index_end][-1].end]
						if index_end < len(sentence.tokens):
							if sentence.tokens[index_end].pos == "NN" or sentence.tokens[index_end].pos == "NNS":
								index_end += 1
								term = sentence.text[sentence.tokens[index_start:index_end][0].start:sentence.tokens[index_start:index_end][-1].end]
						sentence_terms.append((sentence.tokens[index_start:index_end][0].dstart, sentence.tokens[index_start:index_end][-1].dend, str(term.strip())))
				
			if "gowords" in rules:
				go_flag = False
				if str(token.text.encode("utf-8")).strip().lower() in go_words:
					index = sentence.tokens.index(token)
					for word in go_words:
						term = str(sentence.tokens[index-1].text) + " " + word
						if term in reduced_term_list:
							#print term, "---", token.text, "---", sentence.text
							go_flag = True
				if go_flag and index-1 > 0 and index+1 < len(sentence.tokens):
					print "********"
					start = sentence.tokens[index-1:index+1][0].dstart
					end = sentence.tokens[index-1:index+1][-1].dend
					term = sentence.text[sentence.tokens[index-1:index+1][0].start:sentence.tokens[index-1:index+1][-1].end]
					sentence_terms.append((start, end, str(term).strip()))


		if "longterms" in rules: #Add terms that are longer than the ones that exist.
			sentence_entities = [] 
			for s in sentence.entities.elist:
				if s.startswith(esource):
					sentence_entities = [str(x.text.encode("utf-8")) for x in sentence.entities.elist[s]]

			for term in [x[2] for x in sentence_terms] + sentence_entities:
				term_tokens = nltk.word_tokenize(term.strip().lower())
				for token in sentence.tokens:
					if term_tokens[0].lower() == token.text.lower():
						start_index = sentence.tokens.index(token)
						end_index = start_index + len(term_tokens)
						if term_tokens == [str(x.text) for x in sentence.tokens[start_index:end_index]]:
							#Look for bigger term to the left
							if start_index > 0:
								if sentence.tokens[start_index-1].text == "and" and sentence.tokens[end_index-1].text in go_words:
									i = 2
									while "," in sentence.tokens[start_index-i:start_index+1-i][0].text:
										term = " ".join([x.text for x  in sentence.tokens[start_index-1-i:end_index]]).replace(" ,", ",")
										start = sentence.tokens[start_index-1-i:end_index][0].dstart
										end = sentence.tokens[start_index-1-i:end_index][-1].dend
										sentence_terms.append((start, end, str(term.strip())))
										i += 2

							#look for bigger term to the right (bigger than 2)
							if end_index < len(sentence.tokens):
								if sentence.tokens[end_index].text == "," and sentence.tokens[start_index].text in go_words:
									i = 2
									while "," in sentence.tokens[end_index+i:end_index+1+i][0].text:
										term = " ".join([x.text for x  in sentence.tokens[start_index:end_index+1+i]]).replace(" ,", ",")
										start = sentence.tokens[start_index:end_index+1+i][0].dstart
										end = sentence.tokens[start_index:end_index+1+i][-1].dend
										if sentence.tokens[start_index:end_index+1+i][-1].text == ",":
											end -=1
											term = term[:-1]
										sentence_terms.append((start, end, str(term.strip())))
										i += 2
									term = " ".join([x.text for x  in sentence.tokens[start_index:end_index+1+i]]).replace(" ,", ",")
									start = sentence.tokens[start_index:end_index+1+i][0].dstart
									end = sentence.tokens[start_index:end_index+1+i][-1].dend
									if sentence.tokens[start_index:end_index+1+i][-1].text == ",":
										end -=1
										term = term[:-1]
									sentence_terms.append((start, end, str(term.strip())))
							#Check if its a double term ex: "x and y anomalies"
							if start_index > 0:
								if sentence.tokens[start_index-1].text == "and":
									term_flag = False
									tok1 = sentence.tokens[start_index-2:start_index-1][0].text
									tok2 = sentence.tokens[end_index-1].text
									if str(tok2) in go_words:
										for word in go_words:
											join_tok = str(tok1) + " " + word
											if join_tok in reduced_term_list:
												term_flag = True
									if term_flag:
										start = sentence.tokens[start_index-2:end_index][0].dstart
										end = sentence.tokens[start_index-2:end_index][-1].dend
										term = sentence.text[sentence.tokens[start_index-2:end_index][0].start:sentence.tokens[start_index-2:end_index][-1].end]
										sentence_terms.append((start, end, str(term).strip()))

		if "small_ent" in rules: #Remove smaller entities
			smaller_entities = set([])
			for a in sentence_terms:
				for b in sentence_terms_text + entity_tokens:
					if a[2].lower() in b and a[2].lower() != b:
						#logging.info("{}: small entity: {} / big entity: {}".format(sentence.sid, a, b))
						smaller_entities.add(a)
			for x in smaller_entities:
				sentence_terms.remove(x)
				#logging.info("{}, removed smaller entity: {}".format(sentence.sid, x))

		for term in reduced_term_list: #Check if sentence is composed by a single entity
			#print term, str(sentence.text)
			if term.lower() == str(sentence.text).lower():
				#print term
				sentence_terms = []
				start = 0
				end = len(term)
				term = str(sentence.text)					
				sentence_terms.append((start, end, term))

		sentence_terms = list(set(sentence_terms))
		#if len(sentence_terms) > 0:
		#	print sentence.sid, sentence_terms
		return sentence_terms

