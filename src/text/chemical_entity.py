import logging
import re
from text.entity import Entity
from config import config

element_base = {
    # number: name symbol ions
    "H": ["Hydrogen", 1],
    "He": ["Helium", 2],
    "Li": ["Lithium", 3],
    "Be": ["Beryllium", 4],
    "B": ["Boron", 5],
    "C": ["Carbon", 6],
    "N": ["Nitrogen", 7],
    "O": ["Oxygen", 8],
    "F": ["Fluorine", 9],
    "Ne": ["Neon", 10],
    "Na": ["Sodium", 11],
    "Mg": ["Magnesium", 12],
    "Al": ["Aluminum", 13],
    "Si": ["Silicon", 14],
    "P": ["Phosphorus", 15],
    "S": ["Sulfur", 16],
    "Cl": ["Chlorine", 17],
    "Ar": ["Argon", 18],
    "K": ["Potassium", 19],
    "Ca": ["Calcium", 20],
    "Sc": ["Scandium", 21],
    "Ti": ["Titanium", 22],
    "V": ["Vanadium", 23],
    "Cr": ["Chromium", 24],
    "Mn": ["Manganese", 25],
    "Fe": ["Iron", 26],
    "Co": ["Cobalt", 27],
    "Ni": ["Nickel", 28],
    "Cu": ["Copper", 29],
    "Zn": ["Zinc", 30],
    "Ga": ["Gallium", 31],
    "Ge": ["Germanium", 32],
    "As": ["Arsenic", 33],
    "Se": ["Selenium", 34],
    "Br": ["Bromine", 35],
    "Kr": ["Krypton", 36],
    "Rb": ["Rubidium", 37],
    "Sr": ["Strontium", 38],
    "Y": ["Yttrium", 39],
    "Zr": ["Zirconium", 40],
    "Nb": ["Niobium", 41],
    "Mo": ["Molybdenum", 42],
    "Tc": ["Technetium", 43],
    "Ru": ["Ruthenium", 44],
    "Rh": ["Rhodium", 45],
    "Pd": ["Palladium", 46],
    "Ag": ["Silver", 47],
    "Cd": ["Cadmium", 48],
    "In": ["Indium", 49],
    "Sn": ["Tin", 50],
    "Sb": ["Antimony", 51],
    "Te": ["Tellurium", 52],
    "I": ["Iodine", 53],
    "Xe": ["Xenon", 54],
    "Cs": ["Cesium", 55],
    "Ba": ["Barium", 56],
    "La": ["Lanthanum", 57],
    "Ce": ["Cerium", 58],
    "Pr": ["Praseodymium", 59],
    "Nd": ["Neodymium", 60],
    "Pm": ["Promethium", 61],
    "Sm": ["Samarium", 62],
    "Eu": ["Europium", 63],
    "Gd": ["Gadolinium", 64],
    "Tb": ["Terbium", 65],
    "Dy": ["Dysprosium", 66],
    "Ho": ["Holmium", 67],
    "Er": ["Erbium", 68],
    "Tm": ["Thulium", 69],
    "Yb": ["Ytterbium", 70],
    "Lu": ["Lutetium", 71],
    "Hf": ["Hafnium", 72],
    "Ta": ["Tantalum", 73],
    "W": ["Tungsten", 74],
    "Re": ["Rhenium", 75],
    "Os": ["Osmium", 76],
    "Ir": ["Iridium", 77],
    "Pt": ["Platinum", 78],
    "Au": ["Gold", 79],
    "Hg": ["Mercury", 80],
    "Tl": ["Thallium", 81],
    "Pb": ["Lead", 82],
    "Bi": ["Bismuth", 83],
    "Po": ["Polonium", 84],
    "At": ["Astatine", 85],
    "Rn": ["Radon", 86],
    "Fr": ["Francium", 87],
    "Ra": ["Radium", 88],
    "Ac": ["Actinium", 89],
    "Th": ["Thorium", 90],
    "Pa": ["Protactinium", 91],
    "U": ["Uranium", 92],
    "Np": ["Neptunium", 93],
    "Pu": ["Plutonium", 94],
    "Am": ["Americium", 95],
    "Cm": ["Curium", 96],
    "Bk": ["Berkelium", 97],
    "Cf": ["Californium", 98],
    "Es": ["Einsteinium", 99],
    "Fm": ["Fermium", 100],
    "Md": ["Mendelevium", 101],
    "No": ["Nobelium", 102],
    "Lr": ["Lawrencium", 103],
    "Rf": ["Rutherfordium", 104],
    "Db": ["Dubnium", 105],
    "Sg": ["Seaborgium", 106],
    "Bh": ["Bohrium", 107],
    "Hs": ["Hassium", 108],
    "Mt": ["Meitnerium", 109],
    "Ds": ["Darmstadtium", 110],
    "Rg": ["Roentgenium", 111],
    "Cn": ["Copernicium", 112],
    "Uuq": ["Ununquadium", 114],
    "Uuh": ["Ununhexium", 116],
}

amino_acids = {
    'Ala': '',
    'Arg': '',
    'Ans': '',
    'Asp': '',
    'Cys': '',
    'Glu': '',
    'Gln': '',
    'Gly': '',
    'His': '',
    'Ile': '',
    'Leu': '',
    'Lys': '',
    'Met': '',
    'Phe': '',
    'Pro': '',
    'Ser': '',
    'Thr': '',
    'Trp': '',
    'Tyr': '',
    'Val': '',
    'Sec': '',
    'Pyl': '',
}

chem_words = set()
chem_stopwords = set()
# words that may seem like they are not part of named chemical entities but they are
for e in element_base:
    chem_words.add(e.lower())
    chem_words.add(element_base[e][0].lower())

#with open("TermList.txt") as termlist:
#    for l in termlist:
#        chem_words.add(l.strip().lower())

# words that are never part of chemical entities
with open(config.stoplist, 'r') as stopfile:
    for l in stopfile:
        w = l.strip().lower()
        if w not in chem_words and len(w) > 1:
            chem_stopwords.add(w)


class ChemicalEntity(Entity):
    """Chemical entities"""
    def __init__(self, tokens, sid, *args, **kwargs):
        # Entity.__init__(self, kwargs)
        super(ChemicalEntity, self).__init__(tokens, *args, **kwargs)
        self.type = "chemical"
        self.subtype = kwargs.get("subtype")
        self.chebi_id = None
        self.chebi_score = 0
        self.chebi_name = None
        self.sid = sid

    def get_dic(self):
        dic = super(ChemicalEntity, self).get_dic()
        dic["subtype"] = self.subtype
        dic["chebi_id"] = self.chebi_id
        dic["chebi_name"] = self.chebi_name
        dic["ssm_score"] = self.ssm_score
        dic["ssm_entity"] = self.ssm_best_ID
        return dic

    def validate(self, ths, rules, *args, **kwargs):
        """
        Use rules to validate if the entity was correctly identified
        :param rules:
        :return: True if entity does not fall into any of the rules, False if it does
        """
        if "stopwords" in rules:
            # todo: use regex
            words = self.text.split(" ")
            stop = False
            for s in chem_stopwords:
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
            if (self.text[0] == "(" and ")" not in self.text) or (self.text[0] == "[" and "]" not in self.text) or \
                    (self.text[0] == "{" and "}" not in self.text):
                logging.debug("parenthesis %s" % self.text)
                self.dstart += 1
                self.start += 1
                self.text = self.text[1:]

        if "hyphen" in rules and "-" in self.text and all([len(t) > 3 for t in self.text.split("-")]):
            logging.debug("ignored hyphen %s" % self.text)
            return False

        #if all filters are 0, do not even check
        if "ssm" in ths and ths["ssm"] != 0 and self.ssm_score < ths["ssm"] and self.text.lower() not in chem_words:
            #logging.debug("filtered %s => %s" % (self.text,  str(self.ssm_score)))
            return False

        if "alpha" in rules:
            alpha = False
            for c in self.text.strip():
                if c.isalpha():
                    alpha = True
                    break
            if not alpha:
                logging.debug("ignored no alpha %s" % self.text)
                return False

        if "dash" in rules and (self.text.startswith("-") or self.text.endswith("-")):
            logging.debug("excluded for -: {}".format(self.text))
            return False
        return True
