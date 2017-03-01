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
from config.config import hpo_conn as db
#from config.config import florchebi_path
#from config.config import stoplist
import glob

hpodic = "data/hpo.pickle"

if os.path.isfile(hpodic):
    logging.info("loading hpo...")
    hpo = pickle.load(open(hpodic, "rb"))
    loadedhpo = True
    logging.info("loaded hpo dictionary with %s entries", str(len(hpo)))
else:
    hpo = {}
    loadedhpo = False
    logging.info("new hpo dictionary")


def find_hpo_term(term, adjust=0):
    ''' returns tuple (hpoID, hpoTerm, score)
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
                   WHERE name =%s and LENGTH(a.name)>0;"""
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
                        and LENGTH(a.term_synonym)>0""", (term,))
        res = cur.fetchone()
        if res is not None:
            # print "2"
            score = 0.8 + adjust
            match = (str(res[0]), res[2], score)

        else:
            # plural - tb pode ser recursivo
            if len(term) > 0 and term[-1] == 's':
                match = find_hpo_term(term[:-1], -0.1)


    #######################################################
    ##HPO has no descriptor table and no ec (in same table)
    #     terms = '("' + '","'.join(term.split(" ")) + '")'
    #     ## JOins several tables and selects dictint words from term list.
    #     query = """SELECT ((sum(d.ic)/ec)-0.1) as score, e.name, c.term_id, c.id,
    #                       group_concat(d.word separator ','), count(d.id), c.descriptor_type
    #                FROM term e JOIN descriptor3 c ON(c.term_id=e.id) JOIN word2term3 b ON (b.descriptor_id=c.id) 
    #                     JOIN word3 d ON (d.id=b.word_id) JOIN SSM_TermDesc f ON (e.id=f.term_id)
    #                WHERE b.word_id IN (
    #                      SELECT distinct id
    #                      FROM word3
    #                      WHERE word in %s)
    #                GROUP by c.id 
    #                ORDER by score desc 
    #                LIMIT 3;""" % (terms,)
    #     # print "QUERY3", query, adjust
    #     cur.execute(query)
    #     res = cur.fetchone()
    #     if res is not None:
    #         # print "3"
    #         match = (str(res[3]), res[1], float(res[0]))
    #         # print term, match
    ######################################################


    if not match or match[2] < 0.0:
        match = ('0', 'null', 0.0)

    return match

def get_IC():
    cur = db.cursor()
    # check for exact match
    query = """SELECT distinct term_id, info_content
               FROM SSM"""
    cur.execute(query)

    res = cur.fetchall()
    return zip(*res)
    

def exit_handler():
    print 'Saving hpo dictionary...!'
    pickle.dump(hpo, open(hpodic, "wb"))

atexit.register(exit_handler)

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
    pickle.dump(syns, open("data/hpo_synonyms.pickle", 'wb'))
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


#makes changes in each entity. In chebi entity it adds id, name and score.
def add_hpo_mappings(results, source): #same as used in evaluate?
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
                        
                        ########### WHAT THIS DO? ###############3
                        #Need to change to find_hpo_term or find out what florchebi does.
                        hpo_info = hpo_resolution.find_hpo_term(entity.text.encode("utf-8"))
                        entity.hpo_id = hpo_info[0]
                        entity.hpo_name = hpo_info[1]
                        entity.hpo_score = hpo_info[2]
                        # TODO: check for errors (FP and FN)
                        if hpo_info[2] == 0:
                            #logging.info("nothing for %s" % entity.text)
                            not_mapped += 1
                        else:
                            #logging.info("%s => %s %s" % (entity.text, hpo_info[1], hpo_info[2]))
                            mapped += 1
                            total_score += hpo_info[2]
    if mapped == 0:
        mapped = 0.000001
    logging.info("{0} mapped, {1} not mapped, average score: {2}".format(mapped, not_mapped, total_score/mapped))
    return results



def main():
    ''' test resolution method by trying with every CEM on CHEMDNER gold standard
        returns '' if resolution fails
    '''
    parser = OptionParser(usage='Perform HPO resoltion')
    parser.add_option("-f", "--file", dest="file",  action="store", default="hpo_dic.pickle",
                      help="Pickle file to load/store the data")
    parser.add_option("-d", "--dir", action="store", dest="dir", type="string", default=".",
                      help="Corpus directory with hpo mappings to measure SSM between pairs (CHEMDNER format)")
    parser.add_option("--reload", action="store_true", default=False, dest="reload",
                      help="Reload pickle data")
    parser.add_option(
        "--log", action="store", dest="loglevel", type="string", default="WARNING",
        help="Log level")
    parser.add_option(
        "--logfile", action="store", dest="logfile", type="string", default="kernel.log",
        help="Log file")
    parser.add_option("--text", action="store", dest="text", type="string", default="water",
                      help="Text to map to hpo")
    parser.add_option(
        "--datatype", action="store", dest="type", type="string", default="chemdner",
        help="Data type to test (chemdner, patents or ddi)")
    parser.add_option("--action", action="store", dest="action", type="string", default="map",
                      help="test, batch, info, map")
    (options, args) = parser.parse_args()
    numeric_level = getattr(logging, options.loglevel.upper(), None)
    #if not isinstance(numeric_level, int):
    #    raise ValueError('Invalid log level: %s' % loglevel)

    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    logging.basicConfig(level=numeric_level, format='%(asctime)s %(levelname)s %(message)s')

    if options.action == "test":
        termlist = []
        if options.type == 'hpo':
            for file in glob.glob(ann_dir + "/*"):
                with open("hpo_sample/annotations/*", 'r') as annotations:
                    for line in annotations:
                        elements = line.strip().split("\t")
                        oth = elements[1].split(" | ")
                        text = oth[1]
                        if text not in termlist:
                            termlist.append(text)
        i = 0
        count = 0
        errors = 0
        sys.stdout.flush()
        for term in termlist:
            count += 1
            res = find_hpo_term(term, 0)
            if res[1] == 'null':
                i += 1
            elif len(term) > 1:
                if term[1] != res[0]:
                    errors += 1
        print " nulls: " + str(i) + ' errors:' + str(errors) + ' total:' + str(count)

    elif options.action == "info":
        info = get_IC()
        pickle.dump(info, open("hpo_IC.pickle", "wb"))

    elif options.action == "map":
        print find_hpo_term(options.text)
    elif options.action == "synonyms":
        load_synonyms()

if __name__ == "__main__":
    main()
