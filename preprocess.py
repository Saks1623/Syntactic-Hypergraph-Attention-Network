from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from utils import *
import collections
from collections import Counter
from generate_syntectic import *
import random
import numpy as np
import pickle
import json
from nltk import tokenize
from sklearn.utils import class_weight


def read_file(dataset, LDA=True,syn=True):
	
	doc_content_list = []
	doc_sentence_list = []
	f = open('data/' + dataset + '_corpus.txt', 'rb')

	for line in f.readlines():
	    doc_content_list.append(line.strip().decode('latin1'))
	    doc_sentence_list.append(tokenize.sent_tokenize(clean_str_simple_version(doc_content_list[-1], dataset)))
	f.close()


	doc_content_list = clean_document(doc_sentence_list, dataset)

	max_num_sentence = show_statisctic(doc_content_list)

	doc_train_list_original = []
	doc_test_list_original = []
	labels_dic = {}
	label_count = Counter()


	i = 0
	f = open('data/' + dataset + '_labels.txt', 'r')
	lines = f.readlines()
	for line in lines:
	    temp = line.strip().split("\t")
	    if temp[1].find('test') != -1:
	        doc_test_list_original.append((doc_content_list[i],temp[2]))
	    elif temp[1].find('train') != -1:
	        doc_train_list_original.append((doc_content_list[i],temp[2]))
	    if not temp[2] in labels_dic:
	    	labels_dic[temp[2]] = len(labels_dic)
	    label_count[temp[2]] += 1
	    i += 1

	f.close()
	print(label_count)


	word_freq = Counter()
	word_set = set()
	for doc_words in doc_content_list:
		for words in doc_words:
			for word in words:
				word_set.add(word)
				word_freq[word] += 1

	vocab = list(word_set)
	vocab_size = len(vocab)

	vocab_dic = {}
	for i in word_set:
		vocab_dic[i] = len(vocab_dic) + 1

	print('Total_number_of_words: ' + str(len(vocab)))
	print('Total_number_of_categories: ' + str(len(labels_dic)))

	doc_train_list = []
	doc_test_list = []

	for doc,label in doc_train_list_original:
		temp_doc = []
		for sentence in doc:
			temp = []
			for word in sentence:
				temp.append(vocab_dic[word])
			temp_doc.append(temp)
		doc_train_list.append((temp_doc,labels_dic[label]))

	for doc,label in doc_test_list_original:
		temp_doc = []
		for sentence in doc:
			temp = []
			for word in sentence:
				temp.append(vocab_dic[word])
			temp_doc.append(temp)
		doc_test_list.append((temp_doc,labels_dic[label]))

	keywords_dic = {}
	if LDA:
		keywords_dic_original = pickle.load(open('data/' + dataset + '_LDA.p', "rb" ))
	
		for i in keywords_dic_original:
			if i in vocab_dic:
				keywords_dic[vocab_dic[i]] = keywords_dic_original[i]

	syntactic_dic=[]
	if syn:
		doc_content_syntax=clean_document_syntax(doc_sentence_list,dataset)
		syntactic_dic= load_or_generate_hyperedges(dataset,doc_content_syntax, vocab_dic)

	train_set_y = [j for i,j in doc_train_list]
	unique_classes = np.unique(train_set_y)
	class_weights = class_weight.compute_class_weight(
		class_weight='balanced',
		classes=unique_classes,
		y=train_set_y
	)

	print(f"Class weights computed for {len(class_weights)} classes")

	return doc_content_list, doc_train_list, doc_test_list, vocab_dic, labels_dic, max_num_sentence, keywords_dic, syntactic_dic, class_weights


def loadGloveModel(gloveFile, vocab_dic, matrix_len):
	print("Loading Glove Model")
	f = open(gloveFile,'r',encoding='utf-8')
	gloveModel = {}
	glove_embedding_dimension = 0
	for line in f:
		splitLine = line.split()
		word = splitLine[0]
		glove_embedding_dimension = len(splitLine[1:])
		embedding = np.array([float(val) for val in splitLine[1:]])
		gloveModel[word] = embedding
	
	words_found = 0
	weights_matrix = np.zeros((matrix_len, glove_embedding_dimension))
	weights_matrix[0] = np.zeros((glove_embedding_dimension, ))
	
	for word in vocab_dic:
		if word in gloveModel:
			weights_matrix[vocab_dic[word]] = gloveModel[word]
			words_found += 1
		else:
			weights_matrix[vocab_dic[word]] = gloveModel['the']

	print("Total ", len(vocab_dic), " words")
	print("Done.",words_found," words loaded from", gloveFile)

	return weights_matrix

import os
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

def loadBERTModel(vocab_dic, dataset_name):
    fine_tuned_path = f"data/{dataset_name}_bert"
    use_fine_tuned = os.path.exists(fine_tuned_path)

    if use_fine_tuned:
        print(f" Loading fine-tuned BERT model from '{fine_tuned_path}'...")
        bert_tokenizer = BertTokenizer.from_pretrained(fine_tuned_path)
        bert_model = BertModel.from_pretrained(fine_tuned_path)
        
    else:
        print(f" Fine-tuned model not found. Using 'bert-base-uncased'...")
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')

    bert_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bert_model.to(device)
    embedding_dim = model.config.hidden_size

    weights_matrix = torch.zeros((len(vocab_dic) + 1, embedding_dim), device=device)

    for word, idx in tqdm(vocab_dic.items(), desc="Extracting BERT embeddings"):
        tokens = bert_tokenizer.tokenize(word.lower())
        inputs = bert_tokenizer(tokens, return_tensors='pt', is_split_into_words=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        word_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        weights_matrix[idx] = word_embedding

    source = "fine-tuned" if use_fine_tuned else "original"
    print(f" {len(vocab_dic)} words loaded using {source} BERT model.")
    return weights_matrix