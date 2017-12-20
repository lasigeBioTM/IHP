import nltk
import logging
from pycorenlp import StanfordCoreNLP
import glob
import codecs

stopwords = ["all", "high", "left", "lower", "low", "right", "central", "severe", "mild", "large", "onset",
			 "bilateral", "multiple", "profound", "proximal", "distal", "foot", "long", "papules", "middle",
			 "position", "congenital", "inheritance", "small", "genitalia", "brachypodism",
			 "localized", "ear", "prominent", "peripheral", "mosaicism", "fibroma", "arrhythmia", 
			 "hamartoma", "nevus", "neuroma", "severity", "approximately", #schwanomma used several times in some cases.
			  "attenuation", "age of onset", "nevi", "BDC", "unilateral", "neurinoma", "learning",
			 "macrocephaly", "progressive", "soft", "heterozygosity", "speech", "audiometry", "generalized",
			 "stereotyped", "fine", "diffuse", "carbamazepine", "oxcarbazepine", "interview",
			 "frequent", "infections", "wide", "fluorescence", "hybridization", "tumorigenesis", "anemia",
			 "helix", "audiometry", "microduplications", "mitochondrial", "IHH", "irregular",
			 "localised", "mononeuropathy", "naevus", "proliferation", "deletion", "thin", "epicanthal", "kidneys",
			 "broad", "flattened", "echocardiography", "numbness", "asymmetric", "Deletion", "nonspecialty",
			 "microsatellite", "persistent", "metacarpal", "schwannomin", "hybridisation", "secretion", "coronoid", 
			 "episodes", "vocalization", "configuration", "acute", "membrane", "activities", "diagnosis",
			 "ophthalmologic", "nervous system", "acrodysostosis", "carcinoma", "genodermatosis", "many",
			 "hemangiomatous", "infra", "hippocampus", "progeria", "neural crest", "spinal", "neonatal",
			 "awareness", "tumorigenesis", "acrodysostosis", "numbness", "unusual", "mesothelioma", "brain stem",
			 "capsular", "posterior", "subcapsular", "tarsal bones", "left eye", "ear", "ears", "third", "first", "second",
			 "branchial arch", "choroidal", "crossed", "perineal", "clinical findings", "clinical manifestations",
			 "characteristic manifestations", "vestibular studies", "genetic defect", "hypernasal", "buccal smear", "neuroma",
			 "neuromas", "central nervous system", "cervical pits", "normal head circumference",
			 "vocal cords", "movement", "activity", "renal pelvis", "eyebrow", "umbilical cord", "albumin",
			 "pituitary gland", "pineal gland", "apoptosis", "achilles tendon", "cortical cyst", "corneal stroma",
			 "motile cilia", "oral cavity", "bile duct", "renal pelvis", "thymic hormone", "lymphangiectasia",
			 "maxilla", "subcutaneous fat", "extraocular muscle", "bile ducts", "adrenocorticotropic hormone", 
			 "hypermobility", "lacrimal glands", "atresia"]
			  #nonprogressive should also be added even if it reduces score...

removewords = [';', '!', '?', 'author', ' function', ' ability', ' evaluations', ' laboratory',
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
			   'candidate', 'ube3a', 'suppression', ' fly ', 'counterpart', 'meaning', 'doubtful', 'locomotive', 'particularly',
			   'potential ', 'misdiagnosis', 'able to', 'pronounce', 'placement', 'deletional', 'recognisable',
			   'developing', 'enucleation', 'nondisjunction', 'animal', 'show ', 'interphase', 'basic ', 'promotes',
			   'certainty', ' termination', ' normal', 'textbooks', 'cohort', 'counselling', 'dilemma', 'ptch2', 'attributes',
			   'gamete', 'complementation', 'embryogenesis', 'healthy', 'polytomographic', 'offer', 'monitoring', 'drosophila',
			   'consisting', 'xeroderma', 'nf2', 'confined', 'pelvices', 'xenopus', 'medaka', 'six1', 'six5', 'proband',
			   'recombinants', 'histories', 'care', 'optimal', 'the difficulty', 'medical', 'records', 'men1', 'gadolinium',
			   'unresectable', 'body axis', 'minimal involvement', 'biologically', '``', 'classical', 'investigations',
			   'additional', 'sysdrome', 'echocardiography', 'progeria', 'intronic', 'computerized', 'cryptogenic',
			   'radiological', 'unresectable', 'cell-cell', 'transduction', ' lucencies', 'immunotherapy', 'syllabic',
			   'phenobarbital', 'clonazepam', 'haplotype', 'tropomyosin', 'mixed', "imaging tests"] #clash with regulatory (once in gazette)
			   #author, authors #"function" works in conjunction with go_words to remove thWe 
																	  #common use of the word function

adjectives = ['lacrimal', 'outer', 'auditory', 'nervous', 'autosomal', 'craniofacial', 'global', 'dominant', 'hypoplastic',
			  'imperforate', 'facial', 'cortical', 'spinal', 'skeletal', 'acoustic', 'anal', 'mental', 'peripheral', 'abnormal',
			  'genital', 'sporadic', 'developmental', 'preaxial', 'ovarian', 'conductive', 'branchial', 'ocular', 'cervical',
			  'myoclonic', 'central', 'mixed', 'distal', 'subcapsular/capsular', 'ataxic', 'auricular', 'plantar', 'subcapsular',
			  'inner', 'happy', 'capsular', 'retinal', 'intracranial', 'posterior', 'external', 'ophthalmic', 'bilateral', 'congenital',
			  'wide', 'preauricular', 'unilateral', 'middle', 'cardiac', 'multiple', 'palmar', 'vestibular', 'variable', 'basal', 'short',
			  'renal', 'nasolacrimal', 'autosomal-dominant', 'sensorineural', 'severe', 'oral']

good_nouns = ["development ", "morphogenesis", "origin", "features", "nerve", "diagnosis", "structures"] #performance. closure

pre_entity_terms = ["mild to severe", "mild to moderate", "mild to profound",
					"moderate to severe", "moderate to profound", "severe to profound"]

go_words = ["abnormalities", "anomalies","involvement", "abnormality", "anomaly",
			"disturbances", "immaturity", "defects", "absence", "malformation", "malformations", "duplication",
			"problems", "lesions", "syndromes", 'calcification', 'loss', 'delay',
			'episodes', "disorders", "findings", "syndrome", "hypoplasia", "contractures",
			"manifestations", "shortening"] #Need to fix for ocular manifestations for example-

pos_go_words = ["of the", "in the", "of"]
suffixes = ["oma"]

definingwords = ["group", "type"]
non_words = ["in", "to", "of", "the"]


#simple_terms = [] #LDDB List containing terms with only 2 tokens
#a = open("data/LDDB2HPO_terms.txt")
#for line in a:
#	t = line.strip()
#	simple_terms.append(t)
corenlp_client = StanfordCoreNLP('http://localhost:9000')

class Dictionary():
	def __init__(self):
		self.list_of_terms = open("data/gazette.txt").readlines()
		self.term_dic = {}
		annotation_gazette = open("data/annotation_gazette.txt").readlines()

		for x in self.list_of_terms + annotation_gazette:
			term = x.strip().lower()
			tokenized = nltk.word_tokenize(term)
			self.term_dic[tokenized[0]] = []
			self.term_dic[tokenized[-1]] = []

		for x in self.list_of_terms + annotation_gazette:
			term = x.strip().lower()
			tokenized = nltk.word_tokenize(term)
			self.term_dic[tokenized[0]].append(term)
			self.term_dic[tokenized[-1]].append(term)

	def easy_search_terms(self, sentence, esource, ths, rules, off_list=None):
		""" Search new entities using a dictionary. Checks if the
			entities are already checked
			Sentence is an IBEnt Sentence Object"""
		reduced_term_list = []
		reduced_term_dic = {}
		sentence_terms = []
		sentence_terms_text = []
		entity_tokens = []
		aaa = open("aaa2.txt", "a")
		sentence_entities = []

		if "gen_rules" in rules:
			for s in sentence.entities.elist:
				if s.startswith(esource):
					sentence_entities = [str(x.text.encode("utf-8")).lower() for x in sentence.entities.elist[s]]
			#Create a gazette with words that start with any word of the sentence
			#Why was this after next section?
			for token in sentence.tokens:
				ptoken = str(token.text.encode("utf-8")).lower().strip()
			
				if ptoken in self.term_dic:
					for term in self.term_dic[ptoken]:
						if ptoken not in reduced_term_dic:
							reduced_term_dic[ptoken] = []
						reduced_term_dic[ptoken].append(term)
						#aaa.write(term + "\n")
						reduced_term_list.append(term)
						x = str(term).strip()

			reduced_term_list = list(set(reduced_term_list))

			if off_list != None:
				if "twice_validated" in rules:
					to_add = []
					for term in off_list + reduced_term_list:
						tsplit = term.split(" ")
						if tsplit[0] not in reduced_term_dic:
							reduced_term_dic[tsplit[0]] = []
						if tsplit[-1] not in reduced_term_dic:
							reduced_term_dic[tsplit[-1]] = []						
						# if len(tsplit) == 2:
						# 	t = tsplit[1] + " in the " + tsplit[0]
						# 	t2 = tsplit[1] + " of the " + tsplit[0]
						# 	#print sentence.sid, t
						# 	if t in sentence.text.encode("utf-8").lower() or t2 in sentence.text.encode("utf-8").lower():
						# 		#print t
						# 		reduced_term_dic[tsplit[0]].append(tsplit[1] + " in the " + tsplit[0])
						# 		reduced_term_dic[tsplit[0]].append(tsplit[1] + " of the " + tsplit[0])

						if len(tsplit) > 1: #Get linguistic variations of entities
							if tsplit[-1] in go_words:
								for w in pos_go_words:
									reduced_term_dic[tsplit[-1]].append(tsplit[-1] + " " + w + " " + " ".join(tsplit[:-1]))
									if tsplit[-1][-1] == "s":
										if tsplit[-1][:-1] not in reduced_term_dic:
											reduced_term_dic[tsplit[-1][:-1]] = []
										reduced_term_dic[tsplit[-1][:-1]].append(tsplit[-1][:-1] + " " + w + " " + " ".join(tsplit[:-1]))
									else:
										if tsplit[-1]+"s" not in reduced_term_dic:
											reduced_term_dic[tsplit[-1]+"s"] = []	
										reduced_term_dic[tsplit[-1]+"s"].append(tsplit[-1]+"s" + " " + w + " " + " ".join(tsplit[:-1]))
							if tsplit[0] in go_words:
								for w in pos_go_words:
									if tsplit[0] + " " + w in term:
										l = len(w.split(" ")) + 1
										reduced_term_dic[tsplit[0]].append(" ".join(tsplit[len(tsplit)-l:]) + " " + tsplit[0])
										if tsplit[0][-1] == "s":
											reduced_term_dic[tsplit[0]].append(" ".join(tsplit[len(tsplit)-l:]) + " " + tsplit[0][:-1])
										else:
											reduced_term_dic[tsplit[0]].append(" ".join(tsplit[len(tsplit)-l:]) + " " + tsplit[0]+"s")				

						if "type" in term:
							i = term.find("type")
							x = term[:i].strip()
							y = term[i:].strip()
							reduced_term_dic[tsplit[0]].append(x + " " + y)
							reduced_term_dic[tsplit[0]].append(x + " " + y.replace("type", ""))
							reduced_term_dic[tsplit[0]].append(y + " " + x)
							reduced_term_dic[tsplit[0]].append(y + " " + x.replace("type", ""))
						if term[-1] == "s":
							reduced_term_dic[tsplit[0]].append(term[:-1])
						if term[-1] != "s":
							reduced_term_dic[tsplit[0]].append(term+"s")

			for x in reduced_term_dic:
				reduced_term_dic[x] = list(set(reduced_term_dic[x]))

			for i in range(len(sentence.tokens)):

				if "exact" in rules:
				#Exact match recognition versus gazette
					tok = str(sentence.tokens[i].text.encode("utf-8").lower().strip())
					if tok in reduced_term_dic.keys():
						for term in reduced_term_dic[tok]:
							term_tokens = nltk.word_tokenize(term.lower().replace("/", " / ").replace("-", " - ").replace('"', ' " '))
							start_index = i
							end_index = start_index + len(term_tokens) - 1
							entity_tokens = []
							#aaa.write(term + "\n")
							for token in sentence.tokens[start_index:end_index+1]:
								entity_tokens.append(token.text.encode("utf-8").lower())
							if term_tokens == entity_tokens:
								start = sentence.tokens[start_index].dstart
								end = sentence.tokens[end_index].dend
								term = str(sentence.text[sentence.tokens[start_index].start:sentence.tokens[end_index].end])
								#aaa.write(term + "\n")					
								sentence_terms.append((start, end, term))
								sentence_terms_text.append(term)

				if "variation" in rules:
					tok = str(sentence.tokens[i].text.encode("utf-8").lower().strip())
					#Posgowords - Combination of key nouns with phrasal queues
					pos_flag = False
					if tok in go_words:
						tlist = []
						for word in pos_go_words:
							term = tok + " " + word

							if term in str(sentence.text.encode("utf-8")):

								tlist.append(term)
						if len(tlist) > 0: 

							part_term = max(tlist, key=len) #choose longest to prefer "of the" over "of"

							part_term_len = len(part_term.split(" "))
							start_index = i
							end_index = i + part_term_len 
							if end_index+1 < len(sentence.tokens):
								term = sentence.text[sentence.tokens[start_index].start:sentence.tokens[end_index].end]
								#print "****0 {}".format(str(term.strip()))
								if sentence.tokens[end_index+1].pos not in ["NN", "NNS", "NNP", "NNPS"]: #in case there is a word defining the noun
									end_index += 1
									term = sentence.text[sentence.tokens[start_index].start:sentence.tokens[end_index].end]
									if end_index+1 < len(sentence.tokens):
										if sentence.tokens[end_index+1].pos in ["NN", "NNS", "NNP", "NNPS"]:
											end_index += 1
											term = sentence.text[sentence.tokens[start_index].start:sentence.tokens[end_index].end]								
											sentence_terms.append((sentence.tokens[start_index].dstart, sentence.tokens[end_index].dend, str(term.strip())))
											sentence_terms_text.append(term.strip())			
											#print "****1 {}".format(str(term.strip()))
								else:
									term = sentence.text[sentence.tokens[start_index].start:sentence.tokens[end_index+1].end]
									sentence_terms.append((sentence.tokens[start_index].dstart, sentence.tokens[end_index+1].dend, str(term.strip())))
									sentence_terms_text.append(term.strip())								
									#print "****2 {}".format(str(term.strip()))

							#Entities containing entities
							end_index = i + part_term_len 
							if end_index+2 < len(sentence.tokens): #Fix defects of the outer, ad, ad, ad noun			
								if (sentence.tokens[start_index].text in go_words and
									sentence.tokens[end_index].pos in ["JJ", "JJR", "JJS"] and
									sentence.tokens[end_index+1].pos == ","):
									term = sentence.text[sentence.tokens[start_index].start:sentence.tokens[end_index].end]	
									ei = 0
									while "," in sentence.tokens[end_index+ei+1].pos and sentence.tokens[end_index+ei+1].pos in ["JJ", "JJR", "JJS"]:
										ei += 2
									if sentence.tokens[end_index+ei+1].pos == "CC":
										term = " ".join([x.text for x in sentence.tokens[start_index:end_index+4+ei]]).replace(" ,", ",")
										start = sentence.tokens[start_index].dstart
										end = sentence.tokens[end_index+3+ei].dend
										sentence_terms.append((start, end, str(term.strip())))
										sentence_terms_text.append(term.strip())

					#Go Words
					go_flag = False
					if tok in reduced_term_dic.keys():
						if tok in go_words:
							for word in go_words:
								term = str(sentence.tokens[i-1].text.encode("utf-8").lower().strip()) + " " + word
								if term in reduced_term_dic[tok] + sentence_terms_text:
									go_flag = True
						if go_flag and i-1 > 0 and i <= len(sentence.tokens):
							start = sentence.tokens[i-1].dstart
							end = sentence.tokens[i].dend
							term = sentence.text[sentence.tokens[i-1].start:sentence.tokens[i].end]
							sentence_terms.append((start, end, str(term).strip()))
							sentence_terms_text.append(term.strip())

					#Checks if sentence is composed by only one entity - Reduces precision due to corpus annotations
					if tok in reduced_term_dic.keys():
						for term in reduced_term_dic[tok]:
							if tok == term:
								start = sentence.tokens[i].dstart
								end = sentence.tokens[i].dend
								sentence_terms.append((start, end, str(sentence.tokens[i].text.encode("utf-8"))))

				#add words that end in a specific suffix like "oma" - Reduces precision due to corpus annotations
				# if tok[-3:] in suffixes:
				# 	start = sentence.tokens[i].dstart
				# 	end = sentence.tokens[i].dend
				# 	sentence_terms.append((start, end, str(sentence.tokens[i].text.encode("utf-8")).strip()))

				if "longer" in rules:
					tok = str(sentence.tokens[i].text.encode("utf-8").lower().strip())
					#Checks for the existance of longer terms.
					found_entities = {}
					for term in [x[2].lower() for x in sentence_terms] + sentence_entities:
						term_tokens = nltk.word_tokenize(term.lower().replace("/", " / ").replace("-", " - "))
						if term_tokens[0] not in found_entities.keys():
							found_entities[term_tokens[0]] = []
						found_entities[term_tokens[0]].append(term)
					for term in found_entities:
						found_entities[term] = set(list(found_entities[term]))

					if tok in found_entities.keys():
						for term in found_entities[tok]:
							term_tokens = nltk.word_tokenize(term.lower().replace("/", " / ").replace("-", " - "))
							start_index = i
							end_index = start_index + len(term_tokens) - 1
							entity_tokens = []
							if term_tokens == [str(x.text.encode("utf-8")).lower() for x in sentence.tokens[start_index:end_index+1]]:
								#Try to expand to the left
								if start_index - 2 > 0 and start_index+1 < len(sentence.tokens):
									if (sentence.tokens[start_index-1].text.encode("utf-8").lower() == "and" and
									    sentence.tokens[start_index+1].text.encode("utf-8").lower() in go_words and
									    sentence.tokens[start_index-2].text.encode("utf-8").lower() != "," and
									    sentence.tokens[start_index-2].text.encode("utf-8").lower() not in go_words):
										counter = 2
										if start_index-1-counter > 0:
											while "," in sentence.tokens[start_index-1-counter:start_index+1-counter][0].text:
												if start_index-1-counter > 0:
													term = " ".join([x.text for x  in sentence.tokens[start_index-counter:end_index+1]]).replace(" ,", ",")
													start = sentence.tokens[start_index-counter:end_index+1][0].dstart
													end = sentence.tokens[start_index-counter:end_index+1][-1].dend
													sentence_terms.append((start, end, str(term.strip())))
													sentence_terms_text.append(term.strip())
													counter += 2
											term = " ".join([x.text for x  in sentence.tokens[start_index-counter:end_index+1]]).replace(" ,", ",")
											start = sentence.tokens[start_index-counter:end_index+1][0].dstart
											end = sentence.tokens[start_index-counter:end_index+1][-1].dend
											sentence_terms.append((start, end, str(term.strip())))
											sentence_terms_text.append(term.strip())

								#Try to expand to the left
								if start_index - 2 > 0 and start_index+1 < len(sentence.tokens):
									if (sentence.tokens[start_index-1].text.encode("utf-8").lower() == "and" and
									    sentence.tokens[start_index+1].text.encode("utf-8").lower() in go_words and
									    sentence.tokens[start_index-2].text.encode("utf-8").lower() == ","):
									    #sentence.tokens[start_index-2].text.encode("utf-8").lower() not in go_words):
										counter = 2
										if start_index-1-counter > 0:
											while "," == sentence.tokens[start_index-2-counter].text:
												if start_index-2-counter > 0:
													term = " ".join([x.text for x  in sentence.tokens[start_index-counter-1:end_index+1]]).replace(" ,", ",")
													start = sentence.tokens[start_index-counter-1:end_index+1][0].dstart
													end = sentence.tokens[start_index-counter-1:end_index+1][-1].dend
													sentence_terms.append((start, end, str(term.strip())))
													sentence_terms_text.append(term.strip())
													#print term
													counter += 2
											term = " ".join([x.text for x  in sentence.tokens[start_index-counter-1:end_index+1]]).replace(" ,", ",")
											start = sentence.tokens[start_index-counter-1:end_index+1][0].dstart
											end = sentence.tokens[start_index-counter-1:end_index+1][-1].dend
											sentence_terms.append((start, end, str(term.strip())))
											sentence_terms_text.append(term.strip())
											#print term

								#Try to expand to the right - Tends to have a structure of go_word+pos_go_word+adj+","+adj+..+NN
								if end_index + 2 < len(sentence.tokens):
									if (sentence.tokens[start_index].text in go_words and
										sentence.tokens[end_index+1].text == "," and
										"and" not in sentence.text[sentence.tokens[start_index].start:sentence.tokens[end_index+1].end]):
										i = 2
										while "," in sentence.tokens[end_index+i:end_index+1+i][0].text:
											term = " ".join([x.text for x  in sentence.tokens[start_index:end_index+1+i]]).replace(" ,", ",")
											start = sentence.tokens[start_index:end_index+1+i][0].dstart
											end = sentence.tokens[start_index:end_index+1+i][-1].dend
											if sentence.tokens[start_index:end_index+1+i][-1].text == ",":
												end -=1
												term = term[:-1]
											sentence_terms.append((start, end, str(term.strip())))
											sentence_terms_text.append(term.strip())
											i += 2
										term = " ".join([x.text for x  in sentence.tokens[start_index:end_index+1+i]]).replace(" ,", ",")
										start = sentence.tokens[start_index:end_index+1+i][0].dstart
										end = sentence.tokens[start_index:end_index+1+i][-1].dend
										if sentence.tokens[start_index:end_index+1+i][-1].text == ",":
											end -=1
											term = term[:-1]
										sentence_terms.append((start, end, str(term.strip())))
										sentence_terms_text.append(term.strip())						

								#Check if its a double term ex: "x and y anomalies"
								if start_index-2 >= 0:
									if (sentence.tokens[start_index-1].text == "and" and
										sentence.tokens[end_index].text in go_words and
										sentence.tokens[start_index-2].pos in ["NN", "NNS"]):
										term_flag = False
										tok1 = sentence.tokens[start_index-2].text
										tok2 = sentence.tokens[end_index].text

										if str(tok2) in go_words:
											for word in go_words:
												join_tok = str(tok1) + " " + word
											if str(sentence.tokens[start_index].text.encode("utf-8")).lower() in reduced_term_dic.keys():
												if join_tok in reduced_term_dic[str(sentence.tokens[start_index].text.encode("utf-8")).lower()]:
													term_flag = True
										if term_flag:
											start = sentence.tokens[start_index-2:end_index][0].dstart
											end = sentence.tokens[start_index-2:end_index][-1].dend
											term = sentence.text[sentence.tokens[start_index-2:end_index][0].start:sentence.tokens[start_index-2:end_index][-1].end]
											sentence_terms.append((start, end, str(term).strip()))
											sentence_terms_text.append(term.strip())

			#Check if sentence is composed by a single entity
			for x in reduced_term_dic:
				for term in reduced_term_dic[x]: 
					if term.lower() == str(sentence.text.encode("utf-8")).lower():
						sentence_terms = []
						start = 0
						end = len(term)
						term = str(sentence.text.encode("utf-8"))					
						sentence_terms.append((start, end, term))

			sentence_terms = list(set(sentence_terms))
		return sentence_terms
