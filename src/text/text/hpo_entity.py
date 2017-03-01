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
hpo_stopwords = set() 


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


	def validate(self, ths, rules):
		"""
		Use rules to validate if the entity was correctly identified
		:param rules:
		:return: True if entity does not fall into any of the rules, False if it does
		"""
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