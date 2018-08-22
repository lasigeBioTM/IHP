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

def validation(results_file, crf, rules, rulesg):
    """Uses IBEnt to load the 2 corpus, do the training, testing and evaluation.
       It returns the precision and recall for the process"""

    #Runs script for load corpus, train, test and evaluate.

    rulesx = "gen_rules twice_validated exact variation longer andor removal stopwords gen_errors lastwords negcon"
    gen_rules = "andor twice_validated lastwords gen_rules"



    os.system("python src/main.py train --goldstd hpo_train --models models/hpo_train --entitytype hpo --crf {}".format(crf))
    os.system("python src/main.py test --goldstd hpo_test -o pickle data/results_hpo_train --models models/hpo_train --entitytype hpo --crf {}".format(crf))
    if rulesg == True and rules == False:
        os.system("python src/evaluate.py evaluate hpo_test --results data/results_hpo_train --models models/hpo_train --entitytype hpo --rules {}".format(gen_rules))
    elif rules:
        os.system("python src/evaluate.py evaluate hpo_test --results data/results_hpo_train --models models/hpo_train --entitytype hpo --rules andor twice_validated lastwords gen_rules {}".format(rulesx))
    else:
        os.system("python src/evaluate.py evaluate hpo_test --results data/results_hpo_train --models models/hpo_train --entitytype hpo")

    # os.system("python src/main.py train --goldstd hpo_train --models models/hpo_train --log DEBUG --entitytype hpo --crf crfsuite")
    # os.system("python src/main.py test --goldstd hpo_test -o pickle data/results_hpo_train --models models/hpo_train --log DEBUG --entitytype hpo --crf crfsuite")
    # os.system("python src/evaluate.py evaluate hpo_test --results data/results_hpo_train --models models/hpo_train --log DEBUG --rules {}".format(rules))
    # else:
    #getting results
    results = open(results_file).readlines()[:6]
    precision = float(results[4].split(": ")[1])
    recall = float(results[5].split(": ")[1])
    print("Results are in: precision || recall -> ", precision, recall)
    
    
    return precision, recall

def cross_validation(main_directory):
    sets = create_sets("corpora/hpo/all/hpo_corpus_text")
    
    final_res = open("src/other/final_cross.txt", "w")

    #nr -> no rules
    #gr -> general rules
    #   -> rules

    #precision_stan = 0
    #recall_stan = 0
    precision_suite = 0
    recall_suite = 0
    #precision_stan_nr = 0
    #recall_stan_nr = 0
    precision_suite_nr = 0
    recall_suite_nr = 0
    #precision_stan_g = 0
    #recall_stan_g = 0
    precision_suite_g = 0
    recall_suite_g = 0    
    res = open("src/other/log_cross_results.txt", "a")
    res.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format("sx.p", "sx.r", "s.p", "s.r", "cx.p", "cx.r", "c.p", "c.r"))
    res.close()
    for i in range(len(sets)): #already for the whole process -> 10 times
        train_set = []
        test_set = []
        for j in range(len(sets)):
            if j != i:
                train_set = train_set + sets[j]
            if j == i:
                test_set = sets[j]

        res = open("src/other/log_cross_results.txt", "a")
        prepare_files(train_set, test_set, "corpora/hpo/") #puts files in correct folders

        os.system("python src/main.py load_corpus --goldstd hpo_train --log DEBUG")
        os.system("python src/main.py load_corpus --goldstd hpo_test --log DEBUG")

        # #StanfordNER evaluation
        #p_stan, r_stan = validation("data/results_hpo_train_report.txt", "stanford", True, True)
        #precision_stan = precision_stan + p_stan
        #recall_stan = recall_stan + r_stan

        #res.write("Stanford.Rules results for round {}: precision: {}, recall: {} \n".format(str(i), str(p_stan), str(r_stan)))

        # # #No rules
        #p_stan_nr, r_stan_nr = validation("data/results_hpo_train_report.txt", "stanford", False, False)
        #precision_stan_nr = precision_stan_nr + p_stan_nr
        #recall_stan_nr = recall_stan_nr + r_stan_nr

        # # #General rules
        #p_stan_g, r_stan_g = validation("data/results_hpo_train_report.txt", "stanford", False, True)
        #precision_stan_g = precision_stan_g + p_stan_g
        #recall_stan_g = recall_stan_g + r_stan_g

        #res.write("Stanford.NoRules results for round {}: precision: {}, recall: {} \n".format(str(i), str(p_stan_g), str(r_stan_g)))

        # #CRFSuite Evaluation
        p_suite, r_suite = validation("data/results_hpo_train_report.txt", "crfsuite", True, True) 
        precision_suite = precision_suite + p_suite
        recall_suite = recall_suite + r_suite

        #res.write("CRFsuite.Rules results for round {}: precision: {}, recall: {} \n".format(str(i), str(p_suite), str(r_suite)))

        # #No Rules
        #p_suite_nr, r_suite_nr = validation("data/results_hpo_train_report.txt", "crfsuite", False, False) 
        #precision_suite_nr = precision_suite_nr + p_suite_nr
        #recall_suite_nr = recall_suite_nr + r_suite_nr

        #General Rules
        #p_suite_g, r_suite_g = validation("data/results_hpo_train_report.txt", "crfsuite", False, True) 
        #precision_suite_g = precision_suite_g + p_suite_g
        #recall_suite_g = recall_suite_g + r_suite_g
        #res.write("CRFSuite.NoRules results for round {}: precision: {}, recall: {} \n".format(str(i), str(p_suite_g), str(r_suite_g)))
        
        res.write("{}\t{}\n".format(p_suite,  r_suite))#str(p_stan), str(r_stan), str(p_stan_nr), str(r_stan_nr), str(p_stan_g), str(r_stan_g), str(p_suite), str(r_suite), str(p_suite_nr), str(r_suite_nr), str(p_suite_g), str(r_suite_g)))
        #Reset sets for next round

        print(str(len(train_set)), len(test_set))
        train_set = []
        test_set = []
        res.close()

    #precision_stan = precision_stan / 10.0
    #recall_stan = recall_stan / 10.0
    precision_suite = precision_suite / 10.0
    recall_suite = recall_suite / 10.0
    #precision_stan_nr = precision_stan_nr / 10.0
    #recall_stan_nr = recall_stan_nr / 10.0
    #precision_suite_nr = precision_suite_nr / 10.0
    #recall_suite_nr = recall_suite_nr / 10.0
    #precision_stan_g = precision_stan_g / 10.0
    #recall_stan_g = recall_stan_g / 10.0
    #precision_suite_g = precision_suite_g / 10.0
    #recall_suite_g = recall_suite_g / 10.0
    #print precision_stan, recall_stan, precision_stan_nr, recall_stan_nr, precision_stan_g, recall_stan_g, precision_suite, recall_suite, precision_suite_nr, recall_suite_nr, precision_suite_g, recall_suite_g

    #final_res.write("Stanford.General Rules\t{}\t{}\n".format(str(precision_stan_g), str(recall_stan_g)))
    #final_res.write("Stanford.Validation Rules\t{}\t{}\n".format(str(precision_stan), str(recall_stan)))
    #final_res.write("Stanford.NoRules\t{}\t{}\n".format(str(precision_stan_nr), str(recall_stan_nr)))
    #final_res.write("CRFSuite.General Rules\t{}\t{}\n".format(str(precision_suite_g), str(recall_suite_g)))
    final_res.write("CrfSuite.Validation Rules\t{}\t{}\n".format(str(precision_suite), str(recall_suite)))
    #final_res.write("CRFSuite.NoRules\t{}\t{}\n".format(str(precision_suite_nr), str(recall_suite_nr)))

    
    final_res.close()
     

def main():
    cross_validation("corpora/hpo/")


if __name__ == "__main__":
    main()
    
    
    
    