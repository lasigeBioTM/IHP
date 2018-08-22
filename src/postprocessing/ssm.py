#!/usr/bin/env python
import MySQLdb
import sys
from .chebi_resolution import find_chebi_term
import os
import pickle
from optparse import OptionParser
import logging
from config.config import chebi_conn as db
# from config.config import go_conn as dbwebgo

measures = ['resnik', 'simui', 'simgic', 'simgic_hindex', 'simui_hindex']
go_measures = ["resnik_go", "simui_go", "simui_hindex_go", "simgic_hindex_go"]


def add_ssm_score(results, source):
    total = 0
    scores = 0
    for did in results.corpus.documents:
        for sentence in results.corpus.documents[did].sentences:
            for s in sentence.entities.elist:
                if s.startswith(source):
                    sentence.entities.elist[s] = get_ssm(sentence.entities.elist[s], "simui", 0)
                    total += 1
                    scores += sum([e.ssm_score for e in sentence.entities.elist[s]])
                    #for entity in results.corpus[did][sid].elist[s]:
                    #    logging.info("%s %s %s %s" % (entity.text, entity.chebi_name, entity.ssm_score,
                    #                                  entity.ssm_chebi_name))
    if total == 0:
        total = 0.00001
    logging.info("average ssm score: {0}".format(scores/total))
    return results


def resnik(id1, id2):
    cur = db.cursor()
    cur.execute("""SELECT MAX(i.rel_info) 
		FROM graph_path p1, graph_path p2, SSM_TermDesc i 
		WHERE p1.term2_id = %s
		AND p2.term2_id = %s 
		AND p1.term1_id = p2.term1_id 
		AND p1.term1_id = i.term_id;""", (id1, id2))
    #r = cur.store_result()
    res = cur.fetchone()[0]
    #print id1, id2, res
    if res is None:
        res = '0'
    return float(res)


def resnik_go(id1, id2):
    cur = dbwebgo.cursor()
    query = """SELECT MAX(t3.ic) 
		FROM graph_path p1, graph_path p2, term t1, term t2, term t3 
        WHERE t1.acc = '%s' AND t2.acc = '%s' AND t1.id = p1.term2_id AND t2.id = p2.term2_id
		AND p1.term1_id = p2.term1_id 
		AND p1.term1_id = t3.id;"""
    cur.execute(query, (id1, id2))
    #r = cur.store_result()
    res = cur.fetchone()[0]
    #print id1, id2, res
    if res is None:
        res = '0'
    return float(res)

def simui_go(id1, id2):
    cur = dbwebgo.cursor()
    query = """SELECT ( 
			SELECT COUNT(y.id)
			FROM  ( 
				SELECT DISTINCT t3.id 
				FROM graph_path p1, graph_path p2, term t1, term t2, term t3 
                WHERE t1.acc = '%s' AND t2.acc = '%s' AND 
                t1.id = p1.term2_id AND t2.id = p2.term2_id
				AND p1.term1_id=p2.term1_id 
				AND p1.term1_id=t3.id)
			AS y )
		 /( 
		 	SELECT COUNT(x.id)   
		 	FROM (  
                SELECT t2.id   
                FROM graph_path p1, term t1, term t2
                WHERE t1.acc = '%s' AND t1.id = p1.term2_id
                AND p1.term1_id = t2.id
                UNION  
                SELECT t4.id
                FROM graph_path p2, term t3, term t4
                WHERE t3.acc = '%s' AND t3.id = p2.term2_id
                AND p2.term1_id = t4.id) 
		 	AS x )"""
    cur.execute(query, (id1,id2,id1,id2))
    res = cur.fetchone()[0]

    if res is None:
        res = '0'
    return float(res)

def simui_hindex_go(id1, id2, h=4):
    cur = dbwebgo.cursor()
    cur.execute("""SELECT (
            SELECT COUNT(y.ic)
            FROM  (
                SELECT DISTINCT f.id, f.ic
                FROM graph_path p1, graph_path p2, term f, term f1, term f2
                WHERE p1.term2_id = f1.id AND f1.acc = %s AND p2.term2_id = f2.id AND f2.acc = %s AND p1.term1_id=p2.term1_id AND p1.term1_id=f.id AND f.hindex >= %s)
            AS y )
         /(
            SELECT COUNT(x.ic)
            FROM (
                SELECT f1.id, f1.ic
                FROM graph_path p1, term f1, term f3
                WHERE f1.acc = f3.id
                    AND f3.acc = %s
                    AND p1.term1_id = f1.id
                    AND f1.hindex >= %s
                UNION
                SELECT f2.id, f2.ic
                FROM graph_path p2, term f2, term f4
                WHERE f2.acc = f4.id
                    AND f4.acc = %s
                    AND p2.term1_id = f2.id
                    AND f2.hindex >= %s)
            AS x )""", (id1,id2,h,id1,h,id2,h))
    res = cur.fetchone()[0]
    if res is None:
        res = '0'
    return float(res)

def resnik_hindex(id1, id2):
    cur = db.cursor()
    cur.execute("""SELECT MAX(i.hindex_info) 
		FROM graph_path p1, graph_path p2, SSM_TermDesc i 
		WHERE p1.term2_id = %s
		AND p2.term2_id = %s 
		AND p1.term1_id = p2.term1_id 
		AND p1.term1_id = i.term_id;""", (id1, id2))
    #r = cur.store_result()
    res = cur.fetchone()[0]
    #print id1, id2, res
    if res is None:
        res = '0'
    return float(res)


def simui(id1, id2):
    cur = db.cursor()
    cur.execute("""SELECT ( 
			SELECT COUNT(y.rel_info)
			FROM  ( 
				SELECT DISTINCT f.term_id, f.rel_info 
				FROM graph_path p1, graph_path p2, SSM_TermDesc f 
				WHERE p1.term2_id = %s
				AND p2.term2_id = %s 
				AND p1.term1_id=p2.term1_id 
				AND p1.term1_id=f.term_id)
			AS y )
		 /( 
		 	SELECT COUNT(x.rel_info)   
		 	FROM (  
                SELECT f1.term_id, f1.rel_info   
                FROM graph_path p1, SSM_TermDesc f1 
                WHERE p1.term2_id  = %s
                AND p1.term1_id = f1.term_id
                UNION  
                SELECT f2.term_id, f2.rel_info   
                FROM graph_path p2, SSM_TermDesc f2
                WHERE p2.term2_id = %s 
                AND p2.term1_id = f2.term_id) 
		 	AS x )""", (id1,id2,id1,id2))
    res = cur.fetchone()[0]
    if res is None:
        res = '0'
    return float(res)

def simui_hindex(id1, id2, h=4):
    cur = db.cursor()
    cur.execute("""SELECT ( 
            SELECT COUNT(y.rel_info)
            FROM  ( 
                SELECT DISTINCT f.term_id, f.rel_info 
                FROM graph_path p1, graph_path p2, SSM_TermDesc f 
                WHERE p1.term2_id = %s
                AND p2.term2_id = %s 
                AND p1.term1_id=p2.term1_id 
                AND p1.term1_id=f.term_id
                AND f.hindex >= %s)
            AS y )
         /( 
            SELECT COUNT(x.rel_info)   
            FROM (  
                SELECT f1.term_id, f1.rel_info   
                FROM graph_path p1, SSM_TermDesc f1 
                WHERE p1.term2_id  = %s
                AND p1.term1_id = f1.term_id
                AND f1.hindex >= %s
                UNION  
                SELECT f2.term_id, f2.rel_info   
                FROM graph_path p2, SSM_TermDesc f2
                WHERE p2.term2_id = %s 
                AND p2.term1_id = f2.term_id
                AND f2.hindex >= %s) 
            AS x )""", (id1,id2,h,id1,h,id2,h))
    res = cur.fetchone()[0]
    if res is None:
        res = '0'
    return float(res)


def simgic(id1, id2):
    cur = db.cursor()
    cur.execute("""SELECT ( 
			SELECT SUM(y.rel_info)
			FROM  ( 
				SELECT DISTINCT f.term_id, f.rel_info 
				FROM graph_path p1, graph_path p2, SSM_TermDesc f 
				WHERE p1.term2_id = %s
				AND p2.term2_id = %s 
				AND p1.term1_id=p2.term1_id 
				AND p1.term1_id=f.term_id)
			AS y )
		 /( 
		 	SELECT SUM(x.rel_info)   
		 	FROM (  
                SELECT f1.term_id, f1.rel_info   
                FROM graph_path p1, SSM_TermDesc f1 
                WHERE p1.term2_id  = %s
                AND p1.term1_id = f1.term_id
                UNION  
                SELECT f2.term_id, f2.rel_info   
                FROM graph_path p2, SSM_TermDesc f2
                WHERE p2.term2_id = %s
                AND p2.term1_id = f2.term_id) 
		 	AS x )""", (id1,id2,id1,id2))
    res = cur.fetchone()[0]
    if res is None:
        res = '0'
    return float(res)

def simgic_hindex(id1, id2, h=4):
    cur = db.cursor()
    cur.execute("""SELECT ( 
            SELECT SUM(y.rel_info)
            FROM  ( 
                SELECT DISTINCT f.term_id, f.rel_info 
                FROM graph_path p1, graph_path p2, SSM_TermDesc f 
                WHERE p1.term2_id = %s
                AND p2.term2_id = %s 
                AND p1.term1_id=p2.term1_id 
                AND p1.term1_id=f.term_id 
                AND f.hindex >= %s)
            AS y )
         /( 
            SELECT SUM(x.rel_info)   
            FROM (  
                SELECT f1.term_id, f1.rel_info   
                FROM graph_path p1, SSM_TermDesc f1 
                WHERE p1.term2_id  = %s
                AND p1.term1_id = f1.term_id
                AND f1.hindex >= %s
                UNION  
                SELECT f2.term_id, f2.rel_info   
                FROM graph_path p2, SSM_TermDesc f2
                WHERE p2.term2_id = %s
                AND p2.term1_id = f2.term_id
                AND f2.hindex >= %s) 
            AS x )""", (id1,id2,h,id1,h,id2,h))
    res = cur.fetchone()[0]
    if res is None:
        res = '0'
    return float(res)

def simgic_hindex_go(id1, id2, h=4):
    cur = dbwebgo.cursor()
    cur.execute("""SELECT (
            SELECT SUM(y.ic)
            FROM  (
                SELECT DISTINCT f.id, f.ic
                FROM graph_path p1, graph_path p2, term f, term f1, term f2
                WHERE p1.term2_id = f1.id AND f1.acc = %s
                AND p2.term2_id = f2.id AND f2.acc = %s
                AND p1.term1_id=p2.term1_id
                AND p1.term1_id=f.id
                AND f.hindex >= %s)
            AS y )
         /(
            SELECT SUM(x.ic)
            FROM (
                SELECT f1.id, f1.ic
                FROM graph_path p1, term f1, term f3
                WHERE p1.term2_id  = f3.id and f3.acc = %s
                AND p1.term1_id = f1.id
                AND f1.hindex >= %s
                UNION
                SELECT f2.id, f2.ic
                FROM graph_path p2, term f2, term f4
                WHERE p2.term2_id = f4.id and f4.acc = %s
                AND p2.term1_id = f2.id
                AND f2.hindex >= %s)
            AS x )""", (id1,id2,h,id1,h,id2,h))
    res = cur.fetchone()[0]
    if res is None:
        res = '0'
    return float(res)


def get_ontology_id(entity, ontology):
    resid = None
    if ontology == "chebi":
        resid = entity.chebi_id
    elif ontology == "go":
        resid = entity.go_id
    return resid


def get_ssm(entities, measure, ontology="chebi", hindex=4):
    """

    :param entities: list of Entity objects relative to a sentence
    :param measure: semantic similarity measure
    :param ontology: (chebi)
    :param hindex: h-index threshold for some measures
    :return: return results with max SSM for each one
    AT LEAST 2 predictions with chebi
    """
    ssms = {} #{e1ID:{e2ID:ssm, e3ID:ssm}}

    if measure not in measures and measure not in go_measures:
        print('measure not implement: ' + measure)
        sys.exit()
    #check if chebi appears at least twice
    #nchebi = 0
    #for res in entities:
    #    if res.chebi_id != '0':
    #        nchebi += 1
    #if nchebi < 2:
    #    return entities

    #calculate SSM between each valid chebiID
    for i1, res1 in enumerate(entities):
        ssms[i1] = {}
        res1id = get_ontology_id(res1, ontology)
        if res1id != '0' and res1id is not None:
            for i2, res2 in enumerate(entities):
                res2id = get_ontology_id(res2, ontology)
                if res1id != res2id and res2id != '0' and res2id is not None: #skip entities with no mapping and same chebiID
                    if measure == "simui_hindex" or measure == "simgic_hindex":
                        ssm = eval('{0}("{1}", "{2}", h={3})'.format(measure, res1id, res2id, str(hindex)))
                    else:
                        ssm = eval('{0}("{1}", "{2}")'.format(measure, res1id, res2id))
                    ssms[i1][i2] = ssm
    #get max ssm for each chebiID
    for i1, res1 in enumerate(entities):
        res1id = get_ontology_id(res1, ontology)
        res1.ssm_score = 0
        if res1id != '0' and len(ssms[i1]) > 0:
            v = list(ssms[i1].values())
            k = list(ssms[i1].keys())
            max_ssm = (k[v.index(max(v))], max(v)) # bestEID, bestSSM
            res1.ssm_best_ID = get_ontology_id(entities[max_ssm[0]], ontology)
            if ontology == "chebi":
                res1.ssm_best_name = entities[max_ssm[0]].chebi_name
            elif ontology == "go":
                res1.ssm_best_name = entities[max_ssm[0]].go_name
            res1.ssm_best_text = entities[max_ssm[0]].text
            #res1.ssm_score = max_ssm[1]
            res1.ssm_score = max_ssm[1]
        else:
            res1.ssm_best_ID = "0"
            res1.ssm_best_name = ""
            res1.ssm_best_text = ""
            if len(entities) == 1 and res1id != '0':
                if ontology == "chebi":
                    res1.ssm_score = 1
                elif ontology == "go":
                    res1.ssm_score = 1
        entities[i1] = res1
    return entities


def firstCommonIC(c1, c2):
    #return the IC of the first common term between two ChEBI terms
    pass


def termsInCommon(id1, id2):
    cur = db.cursor()
    cur.execute("""SELECT COUNT(i.rel_info) 
		FROM graph_path p1, graph_path p2, SSM_TermDesc i 
		WHERE p1.term2_id = %s
		AND p2.term2_id = %s 
		AND p1.term1_id = p2.term1_id 
		AND p1.term1_id = i.term_id;""", (id1, id2))
    res = cur.fetchone()[0]
    if res is None:
        res = '0'
    return int(res)


def harmonicmeanIC(id1, id2):
    if id1 == 0 or id2 == 0:
        return 0
    cur = db.cursor()
    cur.execute("""SELECT f.rel_info 
				FROM SSM_TermDesc f 
				WHERE f.term_id = %s""", (id1,))
    ic1 = float(cur.fetchone()[0])
    cur = db.cursor()
    cur.execute("""SELECT f.rel_info 
				FROM SSM_TermDesc f 
				WHERE f.term_id = %s""", (id2,))
    ic2 = float(cur.fetchone()[0])
    if ic2 == 0 or ic2 == 0:
        return 0
    return (2*ic1*ic2)/(ic1+ic2)


def main():
    ''' input: measure id1 id2 OR measure term1 term2 (performs chebi resolution)'''
    parser = OptionParser(usage='measure SSM between two ChEBI entities. default is all measures between water and ethanol')
    parser.add_option("-f", "--file", dest="file",  action="store", default="ssm",
                      help="Pickle file to load/store the data")
    parser.add_option("-d", "--dir", action="store", dest="dir", type = "string", default=".",
                      help="Corpus directory with chebi mappings to measure SSM between pairs (CHEMDNER format)")
    parser.add_option("--reload", action="store_true", default=False, dest="reload",
                      help="Reload pickle data")
    parser.add_option("--log", action="store", dest="loglevel", type = "string", default = "WARNING",
                      help="Log level")
    parser.add_option("--logfile", action="store", dest="logfile", type = "string", default = "kernel.log",
                      help="Log file")
    parser.add_option("--chebi1", action="store", dest="c1", type = "string", default="15377",
                      help="ChEBI ID of the first entity")
    parser.add_option("--chebi2", action="store", dest="c2", type = "string", default="16236",
                      help="ChEBI ID of the second entity")
    parser.add_option("--ssmtype", action="store", dest="ssm", type = "string", default="all",
                      help="SSM to use, or all")
    parser.add_option("--action", action="store", dest="action", type = "string", default="measure",
                      help="measure, batch")
    (options, args) = parser.parse_args()
    numeric_level = getattr(logging, options.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % options.loglevel)

    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    logging.basicConfig(level=numeric_level, format='%(asctime)s %(levelname)s %(message)s')

    if options.file + '_' + options.action + '.pickle' in os.listdir(os.getcwd()) and not options.reload:
        print("loading data pickle", options.file + '_' + options.action + '.pickle')
        data = pickle.load(open(options.file + '_' + options.action + '.pickle', 'rb'))
    else:
        data = {}

    measure = options.ssm


    id1 = options.c1
    id2 = options.c2
    if options.action == "measure":
        if measure == 'all':
            for m in measures:
                print(m, eval(m + '("' + id1 + '", "' + id2 + '")'))
        elif measure not in measures and measure not in go_measures:
            print('measure not implement: ' + measure)
            sys.exit()
        else:
            print(measure, id1, id2)
            expr = measure + '("' + id1 + '", "' + id2 + '")'
            print(expr)
            print(eval(expr))

if __name__ == "__main__":
    main()
