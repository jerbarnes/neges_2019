# -*- coding: utf-8 -*-
"""
Author: Salud María Jiménez Zafra
Affiliation: SINAI Research group, Computer Science Department, CEATIC, Universidad de Jaén
  
Description: Subtask B scorer

Last modified: February 13, 2019
"""

import sys
import codecs
import argparse

if __name__ == '__main__':

	# Command line options
	sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer)
	parser = argparse.ArgumentParser(description='NEGES 2019 task - Evaluation script for subtask B:\npython3 scorer_subtaskB [OPTIONS] -g <gold standard> -s <system output>\nThis script evaluates a system output with respect to a gold standard. The two files need to have the same number of lines.', formatter_class=argparse.RawTextHelpFormatter)
	requiredNamed = parser.add_argument_group('required named arguments')
	requiredNamed.add_argument('-g', '--gold', help='Gold standard', required=True)
	requiredNamed.add_argument('-s', '--system', help='System output', required=True)

	args = parser.parse_args()

	system_path = args.system
	gold_path = args.gold

	confusion_matrix = {}
	labels = ('positive', 'negative')
	for l1 in labels:
		for l2 in labels:
			confusion_matrix[(l1, l2)] = 0

	# 1. Read files and get labels
	input_labels = {}
	try:
		with open(system_path, 'r') as input_file:
			for line in input_file.readlines():
				
				try:
					id_file, domain, polarity = line.strip().split('\t')
				except:
					print('Wrong file format: ' + system_path)
					sys.exit(1)
				input_labels[id_file + domain] = polarity
	except OSError as e:
		print(e)
		sys.exit(1)
		
			
	try:
		with open(gold_path, 'r') as gold_file:
			for line in gold_file.readlines():
				try:
					id_file, domain, true_polarity = line.strip().split('\t')
				except:
					print('Wrong file format: ' + gold_path)
					sys.exit(1)
				
				key = id_file + domain
				if key in input_labels.keys():
					proposed_polarity = input_labels[key]
					confusion_matrix[(proposed_polarity, true_polarity)] += 1
				else:
					print('Wrong file format: ' + system_path)
					sys.exit(1)
	except OSError as e:
		print(e)
		sys.exit(1)


	### 2. Calculate evaluation measures
	avgP = 0.0
	avgR = 0.0
	avgF1 = 0.0

	for label in labels:
		denomP = confusion_matrix[(label, 'positive')] + confusion_matrix[(label, 'negative')]
		precision = confusion_matrix[(label, label)]/denomP if denomP > 0 else 0
		
		denomR = confusion_matrix[('positive', label)] + confusion_matrix[('negative', label)]
		recall = confusion_matrix[(label, label)]/denomR if denomR > 0 else 0
		
		denomF1 = precision + recall
		f1 = 2*precision*recall/denomF1 if denomF1 > 0 else 0
		print('\t' + label + ':\tPrecision=' + "{0:.3f}".format(precision) + '\tRecall=' + "{0:.3f}".format(recall) + '\tF1=' + "{0:.3f}".format(f1) + '\n')
		
		avgP  += precision
		avgR  += recall
		avgF1 += f1

	avgP /= 2.0
	avgR  /= 2.0
	avgF1 /= 2.0

	accuracy = (confusion_matrix[('positive','positive')] + confusion_matrix[('negative','negative')]) / (confusion_matrix[('positive','positive')] + confusion_matrix[('negative','negative')] + confusion_matrix[('positive','negative')] + confusion_matrix[('negative','positive')])

	print('\nAvg_Precision=' + "{0:.3f}".format(avgP) + '\tAvg_Recall=' + "{0:.3f}".format(avgR) + '\tAvg_F1=' + "{0:.3f}".format(avgF1) + '\tAccuracy=' + "{0:.3f}".format(accuracy))
	 
