#!/usr/bin/env python
import MySQLdb
import sys
import numpy as np
import config

chebidb = config.chebi_conn

godb = config.go_conn

chebicur = chebidb.cursor()
gocur = godb.cursor()

def hindex(term, ontology):
    """
    Get h-index of term/concept
    :param term: Concept ID
    :param ontology: Name of ontology
    :return: H-index value
    """
    if ontology == "chebi":
        cur = chebicur
    elif ontology == "go":
        cur = gocur
    query = '''
        SELECT *
        FROM graph_path
        WHERE term1_id = %s and distance = 1
    ''' % term
    cur.execute(query)
    
    children = cur.fetchall()
    #print children
    if len(children) == 0:
        return 0
        
    else:
        #get nchildren for each node
        nchildren = []
        for child in children:
            #print child[1]
            query = '''
                SELECT *
                FROM graph_path
                WHERE term1_id = %s and distance = 1
            ''' % child[1]
            cur.execute(query)
            children = cur.fetchall()  
            nchildren.append(len(children))
    
        maxh = len(nchildren)
        for n in range(maxh, -1, -1):
            #elementos com n ou mais filhos
            nlist = [c for c in nchildren if c >= n]
            if len(nlist) >= n:
                break
        return n    
            
def main():
    ontology = sys.argv[1]
    if ontology == "chebi":
        cur = chebicur
        db = chebidb
        maintable = "SSM_TermDesc"
        maincol = "term_id"
    elif ontology == "go":
        cur = gocur
        db = godb
        maintable = "term"
        maincol = "id"
    if sys.argv[2] == "update":
        # Add hindex and hindex_info for each ontology concept
        query = "SELECT %s FROM %s;" % (maincol, maintable)
        cur.execute(query)
        ids = cur.fetchall()
        dist = {}
        for id in ids:
            hi = hindex(id[0], ontology)
            h = round(1.0/(hi+1), 6)
            query = "UPDATE {} SET hindex_info = %s WHERE {} = %s".format(maintable, maincol)
            #print query
            cur.execute(query, (h, id[0]))
            query = "UPDATE {} SET hindex = %s WHERE {} = %s".format(maintable, maincol)
            cur.execute(query, (hi, id[0]))
            db.commit()
            print "Row(s) were updated :" +  str(cur.rowcount) 
            if h in dist:
                dist[h] += 1
            else:
                dist[h] = 1
        print dist

    else:
        print hindex(sys.argv[1]), round(1.0/(hindex(sys.argv[1])+1), 6)
            
if __name__ == "__main__":
    main()
