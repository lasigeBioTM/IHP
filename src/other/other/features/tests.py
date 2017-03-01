import glob
import random
import os
import subprocess
import time
import argparse
import codecs
import logging


def brown_clusters(crf):
    #num_clusters -> param c
    #num_col -> param ncollocs
    #min_occur -> param min-occur
    #crf -> stanford or crfsuite

    final_results = open("src/other/features/brown_clusters.txt", "w") #TSV result, num_clusters, num_colls, min_occur
    num_clusters_values = [82]
    #num_col_values = [3]
    min_occur_values = [3]

    output = open("bin/temp/full_corpus.txt", "w")
    lines = []
    for file in glob.glob("corpora/hpo/all/hpo_corpus_text/" + "*"):
        line = open(file, "r").read()
        lines.append(line)

    for line in lines:
        output.write(str(line) + "\n")

    output.close()

    #subprocess.os.chdir("bin/geniass/")
    #subprocess.call(["bin/geniass/geniass", "bin/temp/full_corpus.txt", "bin/temp/full_corpus_separated.txt"])
    #subprocess.os.chdir("bin/brown-cluster/")
    os.system("(cd bin/geniass/; ./geniass ../temp/full_corpus.txt ../temp/full_corpus_separated.txt)")

    i = 0
    for clu in num_clusters_values:
        #for col in num_col_values:
        for occ in min_occur_values:
            i += 1
            os.system("(cd bin/brown-cluster/; ./wcluster --text ../temp/full_corpus_separated.txt --c {} --min-occur {} --output_dir ../temp/clusters)".format(str(clu), str(occ))) # str(col)  --ncollocs {}
            os.system("cp bin/temp/clusters/paths data/")

            #subprocess.call(["cp", "bin/temp/clusters/paths", "data/"])
            f_measure = get_results(crf)
            logging.info("cluster test {}".format(str(i)))
            final_results.write(str(f_measure) + "\t" + str(clu) + "\t" + str(occ) + "\n")# + str(col) + "\n")

    final_results.close()


def get_results(crf):
    os.system("python src/main.py train --goldstd hpo_train --models models/hpo_train --entitytype hpo --crf {}".format(crf))
    os.system("python src/main.py test --goldstd hpo_test -o pickle data/results_hpo_train --models models/hpo_train --entitytype hpo --crf {}".format(crf))
    os.system("python src/evaluate.py evaluate hpo_test --results data/results_hpo_train --models models/hpo_train --entitytype hpo --rules andor stopwords small_ent twice_validated stopwords gowords posgowords longterms small_len quotes defwords digits lastwords")
    results = open("data/results_hpo_train_report.txt").readlines()[:6]
    precision = float(results[4].split(": ")[1])
    recall = float(results[5].split(": ")[1])
    f_measure = (2.0*precision*recall) / (precision + recall)
    return f_measure

def main():
    #Test Boolean Stanford NER Features
    #os.system("python src/other/features/bool_feature_selection.py")

    #Test Boolean Stanford NER Features
    #os.system("python src/other/features/numerical_feature_selection.py")

    #Test Brown Clustering -> Around 20 hours with current values.
    brown_clusters("crfsuite")

if __name__ == "__main__":
    main()