import argparse
import logging
from rdflib import URIRef, BNode, Literal, ConjunctiveGraph, Namespace
from rdflib.namespace import RDF, RDFS
from rdflib.plugins.sparql import prepareQuery
import time
import pprint
from fuzzywuzzy import process
from config import config
pp = pprint.PrettyPrinter(indent=2)
MIRBASE = Namespace("http://www.mirbase.org/")
class MirbaseDB(object):
    def __init__(self, db_path):
        self.g = ConjunctiveGraph()
        self.path = db_path
        self.choices = set()
        self.labels = {}

    def create_graph(self):
        self.g.open(self.path + "data.rdf", create=True)
        data = self.parse_mirbase(self.path)
        #g = ConjunctiveGraph(store="SPARQLUpdateStore")
        # g.bind()
        mirna_class = URIRef("http://purl.obolibrary.org/obo/SO_0000276")
        for mid in data:
            mirna_instance = URIRef(MIRBASE + data[mid]["acc"])
            self.g.add((mirna_instance, RDF.type, mirna_class))
            label = Literal(data[mid]["name"])
            self.g.add((mirna_instance, RDFS.label, label))
            description = Literal(data[mid]["description"])
            self.g.add((mirna_instance, RDFS.comment, description))
            for p in data[mid]["previous_names"]:
                if p.strip():
                    previous_name = Literal(p)
                    self.g.add((mirna_instance, MIRBASE["previous_acc"], previous_name))
            for mature in data[mid]["mature"]:
                mature_instance = URIRef(MIRBASE + data[mid]["mature"][mature]["acc"])
                self.g.add((mature_instance, RDF.type, mirna_class))
                mature_label = Literal(data[mid]["mature"][mature]["name"])
                self.g.add((mature_instance, RDFS.label, mature_label))
                for mature_p in data[mid]["mature"][mature]["previous_names"]:
                    if mature_p.strip():
                        mature_previous_name = Literal(mature_p)
                        self.g.add((mature_instance, MIRBASE["previous_acc"], mature_previous_name))
                self.g.add((mirna_instance, MIRBASE["stemloopOf"], mature_instance))


    def parse_mirbase(self, mirbase_root):
        mirna_dic = {}
        with open(mirbase_root + "mirna.txt") as mirnas:
            for m in mirnas:
                props = m.strip().split("\t")
                mname = props[2]
                mid = props[0]
                macc = props[1]
                mdesc = props[4]
                mprev = props[3].split(";")
                if int(props[-1]) != 22: # not homo sapiens
                    continue
                mirna_dic[mid] = {}
                mirna_dic[mid]["name"] = mname
                mirna_dic[mid]["acc"] = macc
                mirna_dic[mid]["previous_names"] = mprev
                mirna_dic[mid]["description"] = mdesc
        mature_dic = {}
        with open(mirbase_root + "mirna_mature.txt") as mirnas:
            for m in mirnas:
                props = m.strip().split("\t")
                mname = props[1]
                mid = props[0]
                macc = props[3]
                # mdesc = props[4]
                mprev = props[2].split(";")
                if not mname.startswith("hsa-"): # not homo sapiens
                    continue
                mature_dic[mid] = {}
                mature_dic[mid]["name"] = mname
                mature_dic[mid]["previous_names"] = mprev
                mature_dic[mid]["acc"] = macc
        with open(mirbase_root + "mirna_pre_mature.txt") as mirnas:
            for m in mirnas:
                props = m.strip().split("\t")
                mid, matureid = props[:2]
                if mid in mirna_dic:
                    if "mature" not in mirna_dic[mid]:
                        mirna_dic[mid]["mature"] = {}
                    mirna_dic[mid]["mature"][matureid] = mature_dic[matureid]
        # pp.pprint(mirna_dic)
        return mirna_dic

    def map_label(self, label):
        label = label.lower()
        label = label.replace("microrna", "mir")
        label = label.replace("mirna", "mir")
        if not label.startswith("hsa-"):
            label = "hsa-" + label

        result = process.extractOne(label, self.choices)
        # result = process.extract(label, choices, limit=3)
        """if result[1] != 100:
            print
            print "original:", label.encode("utf-8"), result
            # if label[-1].isdigit():
            #     label += "a"
            # else:
            new_label = label + "-1"
            revised_result = process.extractOne(new_label, self.choices)
            if revised_result[1] != 100:
                new_label = label + "a"
                revised_result = process.extractOne(new_label, self.choices)
            if revised_result[1] > result[1]:
                result = revised_result
                print "revised:", label.encode("utf-8"), result"""

        return result


    def load_graph(self):
        self.g.load(self.path + "data.rdf")
        # print "Opened graph with {} triples".format(len(self.g))
        self.get_label_to_acc()
        self.choices = list(self.labels.keys())

    def get_label_to_acc(self):
        for subj, pred, obj in self.g.triples((None, RDFS.label, None)):
            self.labels[str(obj)] = str(subj)
        for subj, pred, obj in self.g.triples((None, RDFS.label, None)):
            self.labels[str(obj)] = str(subj)

    def save_graph(self):
        self.g.serialize(self.path + "data.rdf", format='pretty-xml')
        print('Triples in graph after add: ', len(self.g))
        self.g.close()

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("action", default="create",  help="Actions to be performed.")
    parser.add_argument("--log", action="store", dest="loglevel", default="WARNING", help="Log level")
    parser.add_argument("--label", action="store", dest="label")
    options = parser.parse_args()

    # set logger
    numeric_level = getattr(logging, options.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % options.loglevel)
    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    logging_format = '%(asctime)s %(levelname)s %(filename)s:%(lineno)s:%(funcName)s %(message)s'
    logging.basicConfig(level=numeric_level, format=logging_format)
    logging.getLogger().setLevel(numeric_level)
    total_time = time.time() - start_time
    logging.info("Total time: %ss" % total_time)
    path = config.mirbase_path


    mirbase = MirbaseDB(path)
    if options.action == "create":
        mirbase.create_graph()
        mirbase.save_graph()
    else:
        mirbase.load_graph()
        if options.action == "map":
            print(mirbase.map_label(options.label))
        elif options.action == "geturi":
            q = prepareQuery('SELECT ?s WHERE { ?s rdfs:label ?label .}', initNs={"rdfs": RDFS })
            l = Literal(options.label)
            for row in mirbase.g.query(q, initBindings={'label': l}):
                print(row)
        else:
            m = URIRef("http://www.mirbase.org/cgi-bin/mirna_entry.pl?acc=MI0017413")
            #for s, p, o in g:
            #    print s, p, o
            mirna_class = URIRef("http://purl.obolibrary.org/obo/SO_0000276")
            for row in mirbase.query('select ?s where { ?s rdf:type [] .}'):
                print(row.s)

if __name__ == "__main__":
    main()
