import os
import logging


bool_features = open("src/other/features/boolean_features.txt").readlines()
log = open("src/other/features/bool_log_features.txt", "w")
good_features = []

#Base file based on IBEnt's file. Still need if its good enough
base_features = """trainFile = models/hpo_train.bilou
serializeTo = models/hpo_train.ser.gz
map = word=0,answer=1

useClassFeature=true
useWord=true
maxNGramLeng=14
entitySubclassification = SBIEO
wordShape=chris2useLC\n"""

base_prop = open("bin/stanford-ner-2015-04-20/base.prop", "w")
base_prop.write(base_features)
base_prop.close()
os.system("bash src/other/features/features_selection.sh")
results = open("data/results_hpo_train_report.txt").readlines()[:6]
precision = float(results[4].split(": ")[1])
recall = float(results[5].split(": ")[1])
best_f_measure = (2.0*precision*recall) / (precision + recall)

for feature in bool_features:
	base_prop = open("bin/stanford-ner-2015-04-20/base.prop", "w")
	base_prop.write(base_features)
	base_prop.write(feature.strip() + "=true\n")

	#Accumlates features: Comment for loop use iterative mode. 
	for feat in good_features:
		base_prop.write(feat.strip() + "=true\n")
	base_prop.close()

	#Test read base.prop
	read = open("bin/stanford-ner-2015-04-20/base.prop").read()
	logging.warning(read)

	os.system("bash src/other/features/features_selection.sh")
    results = open("data/results_hpo_train_report.txt").readlines()[:6]
    precision = float(results[4].split(": ")[1])
    recall = float(results[5].split(": ")[1])
    f_measure = (2.0*precision*recall) / (precision + recall)
	if f_measure > best_f_measure:
		best_f_measure = f_measure
		good_features.append(feature.strip())
		logging.warning(feature.strip() + " | " + str(best_f_measure))
	log.write(str(f_measure) + "\t" + feature.strip() + "\n")
log.close()


final_prop = open("base.prop", "w")
final_prop.write(base_features)
for feature in good_features:
	final_prop.write(feature.strip() + "=true\n")

#In the end it needs to return the base.prop file with the best features, a file with the list of features and 
#a file with the log.