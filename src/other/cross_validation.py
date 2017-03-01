import glob
import random
import os

def create_sets(directory):
    """Uses the file directory with all the files to get the file names of the
       sets and create 10 different sets to use for training/testing"""
       
    file_list = [] #Full file list
    sets = [[],[],[],[],[],[],[],[],[],[]] #Divided file list into sets

    #Gets filenames from the directory and saves to a list
    for file in glob.glob(directory + "/*"):
        #print file.split("\\")[-1]
        file_list.append(file.split("/")[-1])
        
    #Randomly selects the files from the list and divides them into 10 sets
    for i in range(len(file_list)):
        random_file = random.choice(file_list)
        sets[i%10].append(random_file)
        file_list.remove(random_file)
    
    return sets

def prepare_files(train_set, test_set, main_directory):
    """Arranges files and folders according to the train/test sets so that it
       can be used for the validation"""
       
    #Reset folders - maybe make it smaller if rm -a removes files but not folders.
    os.system("rm corpora/hpo/train_corpus/*; rm corpora/hpo/train_ann/*; rm corpora/hpo/test_corpus/*; rm corpora/hpo/test_ann/*;".encode('utf-8'))
    
    #Arranges files for the training sets
    for filename in train_set:
        os.system("cp corpora/hpo/all/hpo_corpus_text/{} corpora/hpo/train_corpus/".format(filename))
        os.system("cp corpora/hpo/all/hpo_corpus_annot/{} corpora/hpo/train_ann/".format(filename))   
        
    #Arranges files for the test set
    for filename in test_set:
        os.system("cp corpora/hpo/all/hpo_corpus_text/{} corpora/hpo/test_corpus/".format(filename))
        os.system("cp corpora/hpo/all/hpo_corpus_annot/{} corpora/hpo/test_ann/".format(filename))

    #Create train annotation file to use as part of gazette (exclude test annotations)
    ann_gaz = open("data/annotation_gazette.txt", "w")
    for file in glob.glob("corpora/hpo/train_ann/*"):
        pmid = file.split("/")[-1]
        annotations = open(file, "r")
        for line in annotations:
            elements = line.strip().split("\t")
                
            off = elements[0].split("::")
            start = off[0][1:]
            end = off[1][:-1]

            oth = elements[1].split(" | ")
            id = oth[0]
            text = oth[1].lower().strip()
            doct = "A"
            ann_gaz.write(text + "\n")
        annotations.close()
    ann_gaz.close()

def validation(results_file):
    """Uses IBEnt to load the 2 corpus, do the training, testing and evaluation.
       It returns the precision and recall for the process"""

    #Runs script for load corpus, train, test and evaluate.

    os.system("python src/main.py load_corpus --goldstd hpo_train --log DEBUG")
    os.system("python src/main.py load_corpus --goldstd hpo_test --log DEBUG")
    os.system("python src/main.py train --goldstd hpo_train --models models/hpo_train --log DEBUG --entitytype hpo --crf crfsuite")
    os.system("python src/main.py test --goldstd hpo_test -o pickle data/results_hpo_train --models models/hpo_train --log DEBUG --entitytype hpo --crf crfsuite")
    os.system("python src/evaluate.py evaluate hpo_test --results data/results_hpo_train --models models/hpo_train --log DEBUG --rules andor stopwords small_ent twice_validated stopwords gowords posgowords longterms small_len quotes defwords digits same_terms lastword")

    #getting results
    results = open(results_file).readlines()[:6]
    precision = float(results[4].split(": ")[1])
    recall = float(results[5].split(": ")[1])
    print "Results are in: precision || recall -> ", precision, recall
    
    
    return precision, recall

def cross_validation(main_directory):
    sets = create_sets("corpora/hpo/all/hpo_corpus_text")
    res = open("cross_results.txt", "w")

    precision = 0
    recall = 0
    
    for i in range(len(sets)): #already for the whole process -> 10 times
        train_set = []
        test_set = []
        for j in range(len(sets)):
            if j != i:
                train_set = train_set + sets[j]
            if j == i:
                test_set = sets[j]

        prepare_files(train_set, test_set, "corpora/hpo/") #puts files in correct folders
        p, r = validation("data/results_hpo_train_report.txt") 
        precision = precision + p
        recall = recall + r
        res.write("results for round {}: precision: {}, recall: {} \n".format(str(i), str(p), str(r)))

        #Reset sets for next round
        print len(train_set), len(test_set)
        train_set = []
        test_set = []

    precision = precision / 10.0
    recall = recall / 10.0
    res.write("\n\nFinal precision & recall: {} & {}".format(str(precision), str(recall)))

    res.close()
     
    return precision, recall

def main():
    cross_validation("corpora/hpo/")

if __name__ == "__main__":
    main()
    
    
    
    