import os
import logging

def base_write(good_features, prop_file):
	""" writes base.prop file with base features and gets the precision
		good_features is dictionary with the good features and the value"""
	#Base file based on IBEnt's file. Still need if its good enough
	base_features = """trainFile = models/hpo_train.bilou
serializeTo = models/hpo_train.ser.gz
map = word=0,answer=1

useClassFeature=true
useWord=true
maxNGramLeng=14
entitySubclassification = SBIEO
wordShape=chris2useLC
useNGrams=true
useNeighborNGrams=true
useWordPairs=true
useSymWordPairs=true
useTypeSeqs=true
useTypeSeqs2=true
usePosition=true
useShapeConjunctions=true\n"""

	base_prop = open(prop_file, "w")
	base_prop.write(base_features)
	for feature in good_features:
		base_prop.write(feature + "=" + str(good_features[feature]) + "\n")
	base_prop.close()
	os.system("bash src/other/features/features_selection.sh")

def get_results():
	results = open("data/results_hpo_train_report.txt").readlines()[:6]
	precision = float(results[4].split(": ")[1])
	recall = float(results[5].split(": ")[1])
	base_f_measure = (2.0*precision*recall) / (precision + recall)
	return base_f_measure

def test_feature(feature, value, good_features):
	"""Writes the feature and value in the base.prop and tests. Returns f-measure"""
	base_write(good_features,"bin/stanford-ner-2015-04-20/base.prop")
	base_prop = open("bin/stanford-ner-2015-04-20/base.prop", "a")
	base_prop.write(feature.strip() + "=" + str(value) + "\n")
	base_prop.close()

	#Test read base.prop - To display in console
	read = open("bin/stanford-ner-2015-04-20/base.prop").read()
	logging.warning(read)

	os.system("bash src/other/features/features_selection.sh")
	results = open("data/results_hpo_train_report.txt").readlines()[:6]
	precision = float(results[4].split(": ")[1])
	recall = float(results[5].split(": ")[1])
	f_measure = (2.0*precision*recall) / (precision + recall)
	return f_measure

def feat_type(type, min_range, max_range, low_value, top_value):
	best_val = 0
	best_f_measure = 0
	for i in range(min_range, max_range):
		val = i
		if type == "non":
			val = low_value + (top_value - low_value)*i*0.1
		feat_f_measure = test_feature(feature, val, good_features)
		logging.warning(feature.strip() + " | " + str(feat_f_measure))
		log.write(feature.strip() + " | " + str(feat_f_measure) + "\n")
		#tests f_measure inside the feature
		if feat_f_measure > best_f_measure:
			best_f_measure = feat_f_measure
			best_val = val

	return best_f_measure, best_val



if __name__ == '__main__':
	numerical_features = open("src/other/features/numerical_features.txt").readlines()
	log = open("src/other/features/num_log_features.txt", "w")
	all_feat = open("src/other/features/all_num_features.txt", "w")
	good_features = {}

	for line in numerical_features:
		feature, type, base_value, low_value, top_value = line.split("\t")
		base_write(good_features,"bin/stanford-ner-2015-04-20/base.prop")
		base_f_measure = get_results()
		if type == "disc":
			best_f_measure, best_val = feat_type(type, int(low_value), int(top_value)+1, int(low_value), int(top_value))
		if type == "non":
			best_f_measure, best_val = feat_type(type, 1, 11, float(low_value), float(top_value))
		all_feat.write(str(best_f_measure) + "\t" + feature + "\t" + str(best_val) + "\n")
		#compares the best feature f_measure with the base to see if improved
	if best_f_measure > base_f_measure:
		good_features[feature] = best_val
		log.write(str(best_f_measure) + "\t" + feature + "\t" + str(best_val) + "\n")

	log.close()
	base_write(good_features,"base.prop")