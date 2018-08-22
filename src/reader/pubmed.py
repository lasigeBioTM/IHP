#!/usr/bin/env python
# -*- coding: utf-8 -*-
import http.client
#import xml.dom.minidom as minidom
#import urllib
import logging
import requests
import time
import sys
import xml.etree.ElementTree as ET
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '../..'))
from text.document import Document
"""
Get texts from PubMed
"""

class PubmedDocument(Document):
    def __init__(self, pmid, **kwargs):
        title, abstract, status = self.get_pubmed_abs(pmid)
        self.abstract = abstract
        super(PubmedDocument, self).__init__(title + "\n" + abstract, ssplit=True, title=title,
                                             did="PMID" + pmid, **kwargs)

    def get_pubmed_abs(self, pmid):
        logging.info("gettting {}".format(pmid))
        #conn = httplib.HTTPConnection("eutils.ncbi.nlm.nih.gov")
        #conn.request("GET", '/entrez/eutils/efetch.fcgi?db=pubmed&id={}&retmode=xml&rettype=xml'.format(pmid))
        payload = {"db": "pubmed", "id": pmid, "retmode": "xml", "rettype": "xml"}
        r = requests.get('http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', payload)
        logging.debug("Request Status: " + str(r.status_code))
        response = r.text

        # logging.info(response)
        title, abstract = self.parse_pubmed_xml(response)
        return title, abstract, str(r.status_code)

    
    def parse_pubmed_xml(self, xml):
        if xml.strip() == '':
            print("PMID not found")
            sys.exit()
        else:
            root = ET.fromstring(xml.encode("utf-8"))
            title = root.find('.//ArticleTitle')
            if title is not None:
                title = title.text
            else:
                title = ""
            abstext = root.findall('.//AbstractText')
            if abstext is not None and len(abstext) > 0:
                abstext = [a.text for a in abstext]
                if all([abst is not None for abst in abstext]):
                    abstext = '\n'.join(abstext)
                else:
                    abstext = ""
            else:
                print("Abstract not found:", title)
                print(xml[:50])
                abstext = ""
                #print xml
                #sys.exit()
        return title, abstext

    
def main():
    pubmeddoc = PubmedDocument(sys.argv[1])
    print(pubmeddoc)
    
if __name__ == "__main__":
    main()
    