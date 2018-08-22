# read transmir database and generate corpus

db_name = "data/transmir_v1.2.tsv"
tfs = set()
diseases = set()
funcs = set()
pmids = set()
mirnas = {} # mirname: (function, disease)
entries = {} # (tfname, mirname): active
with open(db_name, 'r') as dbfile:
    for line in dbfile:
        tsv = line.strip().split("\t")
        if tsv[-1].lower() == "human":
            tfname = tsv[0]
            mirname = tsv[3]
            func = tsv[5].split(";")
            disease = tsv[6].split(";")
            active = tsv[7]
            pmid = tsv[8].split(";")
            tfs.add(tfname.replace("-", "")) # uniform TF names
            for f in func:
                funcs.add(f.strip())
            for d in disease:
                if d != "see HMDD (http://cmbi.bjmu.edu.cn/hmdd)":
                    diseases.add(d.strip())
            for p in pmid:
                pmids.add(p.strip())
            mirnas[mirname] = (func, [d for d in disease if d != "see HMDD (http://cmbi.bjmu.edu.cn/hmdd)"])
            entries[(tfname, mirname)] = (active, pmid)
# print "TF:"
# print tfs
# print "Diseases:"
# print diseases
# print "Functions:"
# print funcs
# print "PMIDs:"
# print pmids
# print "miRNAs:"
# print mirnas.keys()
# print "Entries:"
# print entries
print("Number of TFs: {}".format(len(tfs)))
print("Number of Diseases: {}".format(len(diseases)))
print("Number of Functions: {}".format(len(funcs)))
print("Number of PMIDs: {}".format(len(pmids)))
print("Number of miRNAs: {}".format(len(mirnas)))
print("Number of Entries: {}".format(len(entries)))
with open("corpora/transmir/transmir_pmids.txt", 'w') as f:
    f.write('\n'.join(pmids))
with open("corpora/transmir/transmir_tfs.txt", 'w') as f:
    f.write('\n'.join(tfs))
with open("corpora/transmir/transmir_mirnas.txt", 'w') as f:
    f.write('\n'.join(mirnas))
with open("corpora/transmir/transmir_diseases.txt", 'w') as f:
    f.write('\n'.join(diseases))