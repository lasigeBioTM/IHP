import logging
import re
import MySQLdb
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '../..'))
#from config.config import hpo_conn as db
from text.entity import Entity, Entities
from config import config
from text.token2 import Token2
#from text.offset import Offset, Offsets, perfect_overlap, contained_by

hpo_words = set()
hpo_stopwords = set() #Still to be decided.
#set(["autosomal dominant", "basal cell carcinoma",
 #"basal cell carcinomas", "brachydactyly type C", "cataracts",
 #"sporadic", "acoustic neuromas", "preauricular pits",
 #"severe mental retardation", "distal arthrogryposis"])

#Don't even show up
# syndrome, mutation/s, to, diagnosis, disorder, disease, analysis (doesn't even appear)
# phenotype, 

# with open(config.stoplist, 'r') as stopfile:
#     for l in stopfile:
#         w = l.strip().lower()
#         if w not in prot_words and len(w) > 1:
#             prot_stopwords.add(w)


class HPOEntity(Entity):
	"""HPO entities"""
	def __init__(self, tokens, sid, *args, **kwargs):
		super(HPOEntity, self).__init__(tokens, *args, **kwargs)
		self.type = "hpo"
		self.subtype = kwargs.get("subtype")
		self.nextword = kwargs.get("nextword")
		self.sid = sid
		self.hpo_id = None
		self.hpo_score = 0
		self.hpo_name = 0
        
	#tf_regex = re.compile(r"\A[A-Z]+\d*\w*\d*\Z")

	def get_dic(self):
		dic = super(HPOEntity, self).get_dic()
		#dic["subtype"] = self.subtype
		dic["hpo_id"] = self.hpo_id
		dic["hpo_name"] = self.hpo_name
		dic["ssm_score"] = self.ssm_score
		dic["ssm_entity"] = self.ssm_best_ID
		return dic

	# def write_hpo_line(self, outfile, rank=1):
	# 	if self.sid.endswith(".s0"):
	# 	    ttype = "T"
	# 	else:
	# 	    ttype = "A"
	# 	start = str(self.tokens[0].dstart)
	# 	end = str(self.tokens[-1].dend)
	# 	loc = ttype + ":" + start + ":" + end
	# 	if isinstance(self.score ,dict):
	# 	    conf = sum(self.score.values())/len(self.score)
	# 	else:
	# 	    conf = self.score
	# 	#outfile.write('\t'.join([self.did, loc, str(rank)]) + '\n')
	# 	outfile.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(self.did, loc, str(rank), str(conf), self.text))
	# 	return (self.did, loc, str(rank), str(conf), self.text)


	def validate(self, ths, rules):
		"""
		Use rules to validate if the entity was correctly identified
		:param rules:
		:return: True if entity does not fall into any of the rules, False if it does
		"""
		#a = self.text + "********"
		#logging.info(a)

		# if "andor" in rules:
		# 	words = self.text.split(" ")
		# 	if "and" in words:
		# 		smaller_entity = " ".join(words[:words.index("and")]) 
		# 		logging.info(smaller_entity)
		# 		#add entity to sentence entities list
		# 		#a = len(words) - len(smaller_entity) + 1
		# 		#print self.dend, str(a), str(len(words)), str(len(smaller_entity))
		# 		self.dend -= len(self.text) - len(smaller_entity)
		# 		self.text = smaller_entity
		# 		print self.end
		# 		print self.dend
		# 		print self.text


                # tk_list = []
                # for tk in tlist:
                #     tk_list.append(tk.text)
                # if "and" in tk_list:
                #     smaller_entity = " ".join(tk_list[:tk_list.index("and")])
                #     tk_off = len(newtext) - len(smaller_entity)
                #     ent = create_entity()


		if "stopwords" in rules:
			words = self.text.split(" ")
			#words += self.text.split("-")
			stop = False
			for s in hpo_stopwords:
				if any([s == w.lower() for w in words]):
					logging.debug("ignored stopword %s" % self.text)
					stop = True
			if stop:
				return False

		if "paren" in rules:
			if (self.text[-1] == ")" and "(" not in self.text) or (self.text[-1] == "]" and "[" not in self.text) or \
															      (self.text[-1] == "}" and "{" not in self.text):
				logging.debug("parenthesis %s" % self.text)
				self.dend -= 1
				self.end -= 1
				self.text = self.text[:-1]
		return True

	#Comes from protein_entity
    # def normalize(self):
    #     term = MySQLdb.escape_string(self.text)
    #     # adjust - adjust the final score
    #     match = ()
    #     cur = db.cursor()
    #     # synonym
    #     query = """SELECT DISTINCT t.acc, t.name, s.term_synonym
    #                    FROM term t, term_synonym s
    #                    WHERE s.term_synonym LIKE %s and s.term_id = t.id
    #                    ORDER BY t.ic ASC
    #                    LIMIT 1;""" # or DESC
    #         # print "QUERY", query

    #     cur.execute(query, ("%" + term + "%",))

    #     res = cur.fetchone()
    #     if res is not None:
    #         print res
    #     else:
    #         query = """SELECT DISTINCT t.acc, t.name, p.name
    #                    FROM term t, prot p, prot_GOA_BP a
    #                    WHERE p.name LIKE %s and p.id = a.prot_id and a.term_id = t.id
    #                    ORDER BY t.ic ASC
    #                    LIMIT 1;""" # or DESC
    #         cur.execute(query, (term,))
    #         res = cur.fetchone()
    #         print res


# class HPOEntities(Entities):
#     """Group of entities related to a text"""

#     def __init__(self, **kwargs):
#     	super(HPOEntities, self).__init__(**kwargs)
#         self.elist = {}
#         self.sid = kwargs.get("sid")
#         self.did = kwargs.get("did")

#     def write_hpo_results(self, source, outfile, ths={"ssm":0.0}, rules=[], totalentities=0):
#         """
#         Write results that can be evaluated with the BioCreative evaluation script
#         :param source: Base model path
#         :param outfile: Text Results path to be evaluated
#         :param ths: Thresholds
#         :param rules: Validation rules
#         :param totalentities: Number of entities already validated on this document (for ranking)
#         :return:
#         """
#         lines = []
#         offsets = Offsets()
#         rank = totalentities
#         #    print self.elist.keys()
#         for s in self.elist:
#             #if s != "goldstandard":
#             #    logging.info("%s - %s(%s)" % (self.sid, s, source))
#             if s.startswith(source): #use everything
#                 #logging.info("%s - %s" % (self.sid, s))

#                 for e in self.elist[s]:
#                     val = e.validate(ths, rules)
#                     if not val:
#                         continue

#                     # Overlap rules
#                     eid_offset = Offset(e.dstart, e.dend, text=e.text, sid=e.sid)
#                     exclude = [perfect_overlap]
#                     if "contained_by" in rules:
#                         exclude.append(contained_by)
#                     toadd, v, alt = offsets.add_offset(eid_offset, exclude_if=exclude)
#                     if toadd:
#                         #logging.info("added %s" % e)
#                         line = e.write_hpo_line(outfile, rank)
#                         lines.append(line)
#                         rank += 1
#         return lines, rank

#     def combine_entities(self, base_model, name):
#         """
#         Combine entities from multiple models starting with base_model into one module named name
#         :param base_model: string corresponding to the prefix of the models
#         :param name: new model path
#         """
#         combined = {}
#         offsets = Offsets()
#         for s in self.elist:
#             #logging.info("%s - %s" % (self.sid, s))
#             if s.startswith(base_model) and s != name: #use everything
#                 for e in self.elist[s]: # TODO: filter for classifier confidence
#                     #if any([word in e.text for word in self.stopwords]):
#                     #    logging.info("ignored stopword %s" % e.text)
#                     #    continue
#                     #eid_alt =  e.sid + ":" + str(e.dstart) + ':' + str(e.dend)
#                     next_eid = "{0}.e{1}".format(e.sid, len(combined))
#                     eid_offset = Offset(e.dstart, e.dend, text=e.text, sid=e.sid, eid=next_eid)
#                     added = False
#                     # check for perfect overlaps
#                     for i, o in enumerate(offsets.offsets):
#                         overlap = eid_offset.overlap(o)
#                         if overlap == perfect_overlap:
#                             combined[o.eid].recognized_by.append(s)
#                             combined[o.eid].score[s] = e.score
#                             combined[o.eid].ssm_score_all[s] = e.ssm_score
#                             added = True
#                             #logging.info(combined[o.eid].ssm_score_all)
#                             #logging.info("added {0}-{1} to entity {2}".format(s.split("_")[-1], e.text, combined[o.eid].text))
#                             break
#                     if not added:
#                         offsets.offsets.add(eid_offset)
#                         e.recognized_by = [s]
#                         e.score = {s: e.score}
#                         e.ssm_score_all= {s: e.ssm_score}
#                         combined[next_eid] = e
#                         #logging.info("new entity: {0}-{1}".format(s.split("_")[-1], combined[next_eid].text))
#         self.elist[name] = combined.values()


# class HPOAnnotation(HPOEntity):
# 	""" HPO entity annotated on the HPO Corpus"""
# 	def __init__(self, tokens, sid, **kwargs):
# 		super(HPOAnnotation, self).__init__(tokens, **kwargs)
# 		self.sid = sid