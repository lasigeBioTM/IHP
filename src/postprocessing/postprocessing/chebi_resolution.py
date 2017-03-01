#!/usr/bin/env python
from __future__ import division, unicode_literals
import MySQLdb
import re
import sys
import xml.etree.ElementTree as ET
import os
import shutil
from subprocess import Popen, PIPE
from optparse import OptionParser
import cPickle as pickle
import logging
from sys import platform as _platform
import atexit
from config.config import chebi_conn as db
from config.config import florchebi_path
from config.config import stoplist

chebidic = "data/chebi_dic.pickle"

if os.path.isfile(chebidic):
    logging.info("loading chebi...")
    chebi = pickle.load(open(chebidic, "rb"))
    loadedchebi = True
    logging.info("loaded chebi dictionary with %s entries", str(len(chebi)))
else:
    chebi = {}
    loadedchebi = False
    logging.info("new chebi dictionary")


def find_chebi_term(term, adjust=0):
    ''' returns tuple (chebiID, chebiTerm, score)
        if resolution fails, return ('0', 'null', 0.0)
    '''
    # print "TERM", term
    term = MySQLdb.escape_string(term)
    # adjust - adjust the final score
    match = ()
    cur = db.cursor()
    # check for exact match
    query = """SELECT distinct id, name
                   FROM term a 
                   WHERE name =%s and LENGTH(a.name)>0 and star=3;"""
    # print "QUERY", query
    cur.execute(query, (term,))

    res = cur.fetchone()
    if res is not None:
        # print "1"
        score = 1.0 + adjust
        match = (str(res[0]), res[1], score)
    else:
        # synonyms
        cur.execute("""SELECT a.term_id, a.term_synonym, b.name
                       FROM term_synonym a, term b
                       WHERE a.term_synonym=%s
                        and b.id=a.term_id
                        and LENGTH(a.term_synonym)>0
                        and star=3;""", (term,))
        res = cur.fetchone()
        if res is not None:
            # print "2"
            score = 0.8 + adjust
            match = (str(res[0]), res[2], score)

        else:
            # plural - tb pode ser recursivo
            if len(term) > 0 and term[-1] == 's':
                match = find_chebi_term(term[:-1], -0.1)
#
    if not match:
        #(1)H -> hydrogen-1
        terms = re.sub(r'[\(|\)|\[|\]| ]', ' ',  term)
        termlist = terms.strip().split(" ")
        if len(termlist) == 2:
            if termlist[0].isdigit() and termlist[1] in element_base:
                match = find_chebi_term(element_base[termlist[1]][0] + "-" + termlist[0], -0.1)
            if termlist[1] == '+' and termlist[0] in element_base:
                match = find_chebi_term(element_base[termlist[0]][0] + ' cation', -0.1)
            if termlist[1] == '-' and termlist[0] in element_base:
                match = find_chebi_term(element_base[termlist[0]][0] + ' anion', -0.1)
            # if match != '':
            #    print term, match

    if not match:
        # partial match

        terms = '("' + '","'.join(term.split(" ")) + '")'
        query = """ 
        SELECT ((sum(d.ic)/ec)-0.1) as score, e.name, c.term_id, c.id,
                group_concat(d.word separator ','), count(d.id), c.descriptor_type
        FROM term e JOIN descriptor3 c ON(c.term_id=e.id) JOIN word2term3 b ON (b.descriptor_id=c.id) 
             JOIN word3 d ON (d.id=b.word_id) JOIN SSM_TermDesc f ON (e.id=f.term_id)
        WHERE b.word_id IN
           (SELECT distinct id
            FROM word3
            WHERE word in %s)
        GROUP by c.id 
        ORDER by score desc 
        LIMIT 3;""" % (terms,)
        # print "QUERY3", query, adjust
        cur.execute(query)
        res = cur.fetchone()

        if res is not None:
            # print "3"
            match = (str(res[3]), res[1], float(res[0]))
            # print term, match

    if not match or match[2] < 0.0:
        match = ('0', 'null', 0.0)

    return match


def find_chebi_term2(term):
    if _platform == "linux" or _platform == "linux2":
        # linux
        cp = "{0}/florchebi.jar:{0}/mysql-connector-java-5.1.24-bin.jar:{0}/Tokenizer.jar".format(florchebi_path)
    elif _platform == "win32":
        # "Windows..."
        cp = "{0}/florchebi.jar;{0}/mysql-connector-java-5.1.24-bin.jar;{0}/Tokenizer.jar".format(florchebi_path)
    florcall = ["java", "-cp", cp, "xldb.flor.match.FlorTextChebi3star", db.escape_string(term),
                "children", "true", "mychebi201301", "false", "false", "chebi", stoplist, "1"]
    # print ' '.join(florcall)
    flor = Popen(florcall, stdout=PIPE)
    florresult, error = flor.communicate()
    chebires = florresult.strip().split('\t')
    # print "chebires: ", chebires
    if len(chebires) == 3:
        return (chebires[0], chebires[1], float(chebires[2]))
    else:
        return ('0', 'null', 0.0)


def get_IC():
    cur = db.cursor()
    # check for exact match
    query = """SELECT distinct term_id, rel_info, hindex_info, seco_info
               FROM SSM_TermDesc"""
    # print "QUERY", query
    cur.execute(query)

    res = cur.fetchall()
    return zip(*res)


def find_chebi_term3(term):
    global chebi
    # first check if a chebi mappings dictionary is loaded in memory
    if term in chebi:
        c = chebi[term]
        # chebi mappings are not loaded, or this text is not mapped yet, so update chebi dictionary
    else:
        c = find_chebi_term2(term)
        chebi[term] = c
        logging.info("mapped %s to %s", term.decode("utf-8"), c)
    return c

def exit_handler():
    print 'Saving chebi dictionary...!'
    pickle.dump(chebi, open(chebidic, "wb"))

atexit.register(exit_handler)


def chebi2go(chebiid, go2chebi="chebi2go.pickle"):
    # return a list of GO terms associated with chebi or 0 if none
    mappings = pickle.load(open("data/chebi2go.pickle"))
    if "CHEBI:" + str(chebiid) not in mappings:
        return 0
    else:
        return mappings["CHEBI:" + str(chebiid)]


def loadC2G(go2chebi="GO_to_ChEBI.obo", outname="chebi2go.pickle"):
    # creates a dictionary that maps a ChEBI term to a list of GO terms
    with open(go2chebi, 'r') as c2g:
        lines = c2g.readlines()

    gos = []
    chebis = []
    c2gdic = {}
    for l in lines:
        words = l.strip().split(' ')
        if words[0] == 'id:':
            gos.append(words[1])
            lastgo = words[1]
        elif len(words) > 2 and words[2].startswith("CHEBI"):
            chebis.append(words[2])
            if words[2] not in c2gdic:
                c2gdic[words[2]] = []
            c2gdic[words[2]].append(lastgo)

    counts = {}
    for c in c2gdic:
        if len(c2gdic[c]) not in counts:
            counts[len(c2gdic[c])] = 0
        counts[len(c2gdic[c])] += 1
    print counts
    print "CHEBI:17790 - ", c2gdic["CHEBI:17790"]
    print "gos and chebis:", len(gos), len(chebis)
    print "unique:", len(set(gos)), len(set(chebis))
    pickle.dump(c2gdic, open(outname, 'w'))
    print "mappings written to", outname


def get_description(id):
    cur = db.cursor()
    query = """SELECT term_definition
           FROM term_definition
           WHERE term_id = %s""" % id
    cur.execute(query)
    res = cur.fetchone()

    if res is not None:
        return res[0]
    else:
        return "NA"


def load_synonyms():
    syns = []
    cur = db.cursor()
    query = """SELECT id, name
           FROM term """
    cur.execute(query)
    ids = cur.fetchall()
    for i in ids:
        print "getting synonyms for" + i[1].lower() + '(' + str(i[0]) + ')',
        synset = set()
        synset.add(i[1].lower())
        query = """SELECT term_synonym
           FROM term_synonym
           WHERE term_id = %s""" % i[0]
        cur.execute(query)
        names = cur.fetchall()
        print len(names)
        for name in names:
            #print name[0], 
            synset.add(name[0].lower())
        syns.append(synset)
    pickle.dump(syns, open("data/chebi_synonyms.pickle", 'wb'))
    print "done"


def check_dist_between(cid1, cid2):
    cur = db.cursor()
    query = """SELECT distance 
               FROM graph_path 
               WHERE term1_id = %s and term2_id = %s""" % (cid1, cid2)
    cur.execute(query)
    res = cur.fetchone()
    if res is None:
        dist = -1
    else:
        dist = int(res[0])
    return dist


def add_chebi_mappings(results, source):
    """

    :param results: ResultsNER object
    :return:
    """
    mapped = 0
    not_mapped = 0
    total_score = 0
    for did in results.corpus.documents:
        for sentence in results.corpus.documents[did].sentences:
            for s in sentence.entities.elist:
                if s.startswith(source):
                    #if s != source:
                    #    logging.info("processing %s" % s)
                    for entity in sentence.entities.elist[s]:
                        chebi_info = find_chebi_term3(entity.text.encode("utf-8"))
                        entity.chebi_id = chebi_info[0]
                        entity.chebi_name = chebi_info[1]
                        entity.chebi_score = chebi_info[2]
                        # TODO: check for errors (FP and FN)
                        if chebi_info[2] == 0:
                            #logging.info("nothing for %s" % entity.text)
                            not_mapped += 1
                        else:
                            #logging.info("%s => %s %s" % (entity.text, chebi_info[1], chebi_info[2]))
                            mapped += 1
                            total_score += chebi_info[2]
    if mapped == 0:
        mapped = 0.000001
    logging.info("{0} mapped, {1} not mapped, average score: {2}".format(mapped, not_mapped, total_score/mapped))
    return results



def main():
    ''' test resolution method by trying with every CEM on CHEMDNER gold standard
        returns '' if resolution fails
    '''
    parser = OptionParser(usage='Perform ChEBI resoltion')
    parser.add_option("-f", "--file", dest="file",  action="store", default="chebi_dic.pickle",
                      help="Pickle file to load/store the data")
    parser.add_option("-d", "--dir", action="store", dest="dir", type="string", default=".",
                      help="Corpus directory with chebi mappings to measure SSM between pairs (CHEMDNER format)")
    parser.add_option("--reload", action="store_true", default=False, dest="reload",
                      help="Reload pickle data")
    parser.add_option(
        "--log", action="store", dest="loglevel", type="string", default="WARNING",
        help="Log level")
    parser.add_option(
        "--logfile", action="store", dest="logfile", type="string", default="kernel.log",
        help="Log file")
    parser.add_option("--text", action="store", dest="text", type="string", default="water",
                      help="Text to map to ChEBI")
    parser.add_option(
        "--datatype", action="store", dest="type", type="string", default="chemdner",
        help="Data type to test (chemdner, patents or ddi)")
    parser.add_option("--action", action="store", dest="action", type="string", default="map",
                      help="test, batch, info, map, chebi2go")
    (options, args) = parser.parse_args()
    numeric_level = getattr(logging, options.loglevel.upper(), None)
    #if not isinstance(numeric_level, int):
    #    raise ValueError('Invalid log level: %s' % loglevel)

    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    logging.basicConfig(level=numeric_level, format='%(asctime)s %(levelname)s %(message)s')

    if options.action == "test":
        chemlist = {}
        if options.type == 'chemdner':
            with open("CHEMDNER/CHEMDNER_DEV_TRAIN_V02/chemdner_ann.txt", 'r') as chems:
                for line in chems:
                    tsv = line.strip().split('\t')
                    if tsv[5] not in chemlist:
                        chemlist[tsv[5]] = []
                    chemlist[tsv[5]].append((tsv[4]))

        elif options.type == 'patents':
            for f in os.listdir("patents_corpus/PatentsGoldStandardEnriched"):
                tree = ET.parse("patents_corpus/PatentsGoldStandardEnriched/" + f)
                root = tree.getroot()
                for chem in re.findall('.//ne'):
                    type = chem.get('type')
                    if type not in chemlist:
                        chemlist[type] = []
                    if chem.get("chebi-id") != '':
                        chemlist[type].append(
                            (chem.get("name"), chem.get("chebi-id").split(':')[1]))

        for type in chemlist:
            i = 0
            count = 0
            errors = 0
            print type
            sys.stdout.flush()
            for chem in chemlist[type]:
                count += 1
                res = find_chebi_term(chem[0], 0)
                if res[1] == 'null':
                    i += 1
                elif len(chem) > 1:
                    if chem[1] != res[0]:
                        errors += 1
            print type + " nulls: " + str(i) + ' errors:' + str(errors) + ' total:' + str(count)
    elif options.action == 'batch':

        dir = options.dir
        files = os.listdir(dir)
        for f in files:
            lines = []
            if not f.endswith('with-chebi.txt') and not f.endswith('with-ssm.txt') and\
                                    f + '-with-chebi.txt' not in files and not os.path.isdir(dir + '/' + f):
                print dir + '/' + f
                with open(dir + '/' + f, 'r') as predfile:
                    for line in predfile:
                        tsv = line.strip().split('\t')
                        chebires = find_chebi_term3(tsv[3])
                        lines.append(
                            '\t'.join(tsv[:4]) + '\t' + chebires[0] + '\t' + str(chebires[2]) + '\n')
                with open(dir + '/' + f + '-with-chebi.txt', 'w') as chebifile:
                    for line in lines:
                        chebifile.write(line)
        #print "writing pickle..."
        #pickle.dump(chebidic, open("chebi_dic.pickle", "wb"))
        print "done."

    elif options.action == "info":
        info = get_IC()
        pickle.dump(info, open("chebi_IC.pickle", "wb"))

    elif options.action == "map":
        print find_chebi_term(options.text)
    elif options.action == "chebi2go":
        if options.text == "all":
            loadC2G()
        else:
            print chebi2go(options.text)
    elif options.action == "synonyms":
        load_synonyms()

if __name__ == "__main__":
    main()
