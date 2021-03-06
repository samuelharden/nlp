import os
import re
import pickle
import nltk
import numpy as np
import datetime
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
stop = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import json
from pprint import pprint

# Noun Part of Speech Tags used by NLTK
# More can be found here
# http://www.winwaed.com/blog/2011/11/08/part-of-speech-tags/
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']

def clean_document(document):
    """Cleans document by removing unnecessary punctuation. It also removes
    any extra periods and merges acronyms to prevent the tokenizer from
    splitting a false sentence

    """
    # Remove all characters outside of Alpha Numeric
    # and some punctuation
    document = re.sub('[^A-Za-z .-]+', ' ', document)
    document = document.replace('-', '')
    document = document.replace('...', '')
    document = document.replace('...', '')
    document = document.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs')

    # Remove Ancronymns M.I.T. -> MIT
    # to help with sentence tokenizing
    document = merge_acronyms(document)

    document = document.lower()
    document = remove_stop_words(document)

    # Remove extra whitespace
    document = ' '.join(document.split())
    return document

def remove_stop_words(document):
    """Returns document without stop words"""
    document = ' '.join([i for i in document.split() if i not in stop])
    return document

def merge_acronyms(s):
    """Merges all acronyms in a given sentence. For example M.I.T -> MIT
    """
    r = re.compile(r'(?:(?<=\.|\s)[A-Z]\.)+')
    acronyms = r.findall(s)
    for a in acronyms:
        s = s.replace(a, a.replace('.',''))
    return s

if __name__ == '__main__':
    # Load corpus data used to train the TF-IDF Transformer
    data = pickle.load(open('data.pkl', 'rb'))
    data = [clean_document(da) for da in data]
    train_data = set(data)
    #print train_data
    # Fit and Transform the term frequencies into a vector
    count_vect = CountVectorizer()
    count_vect = count_vect.fit(train_data)
    freq_term_matrix = count_vect.transform(train_data)
    feature_names = count_vect.get_feature_names()

    # Fit and Transform the TfidfTransformer
    tfidf = TfidfTransformer(norm="l2")
    tfidf.fit(freq_term_matrix)
    # Merge corpus data and new document data
    files = os.listdir('../..//processed_papers/')
    abstract = ''
    body = ''
    json_array = {}
    for file in files:
        json_internal_array = {}
	json_internal_array['document_id'] = file
	json_internal_array['matched_sentences'] = {}
	abstract = ''
	sections = ''
	abs_sentences = {}
	body_sentences = {}
	body = ''
        print(file)
        json_data = json.load(open('../../processed_papers/'+file))
        if json_data["abstract_sentences"] is not None:
	    abs_sentences = json_data["abstract_sentences"]
        if json_data["body_sentences"] is not None:
	    body_sentences = json_data["body_sentences"]
	if (len(abs_sentences) == 0) or (len(body_sentences) == 0):
            json_array[file] = json_internal_array
	    continue
        for idx,section in body_sentences.items():
            if section is not None:
		body += " "
                body += section
        title =  abstract
        document =  body
	abs_sentences_value = list(abs_sentences.values())
	abs_sentences_value = [ clean_document(abs_value) for abs_value in abs_sentences_value]
        abs_freq_term_matrix = count_vect.transform(abs_sentences_value)
	abs_sentences_idx = list(abs_sentences.keys())
        abs_tfidf_matrix = tfidf.transform(abs_freq_term_matrix)
        #abs_dense = abs_tfidf_matrix.todense()
	body_sentences_value = list(body_sentences.values())
	body_sentences_value = [ clean_document(body_value) for body_value in body_sentences_value]
        body_freq_term_matrix = count_vect.transform(body_sentences_value)
	body_sentences_idx = list(body_sentences.keys())
        body_tfidf_matrix = tfidf.transform(body_freq_term_matrix)
        #body_dense = body_tfidf_matrix.todense()
        resulting = abs_tfidf_matrix.dot(body_tfidf_matrix.T)
        i = 0
        for result in resulting:
	    #print "Abs_Sent" + abs_sentences[i] + " Body Sentence " + body_sentences[result.argmax()]
            json_internal_array['matched_sentences'][abs_sentences_idx[i]] = body_sentences_idx[result.argmax()]
            i = i + 1
        json_array[file] = json_internal_array
    #story_freq_term_matrix = count_vect.transform([doc])
    #story_tfidf_matrix = tfidf.transform(story_freq_term_matrix)
    #story_dense = story_tfidf_matrix.todense()
    #for sent in abs_sentences:
        # Get Top Ranking Sentences and join them as a summary
    #    top_sents = rank_sentences(doc, sent, doc_matrix, feature_names)
     #   summary = '.'.join([cleaned_document.split('.')[i]
     #                   for i in [pair[0] for pair in top_sents]])
     #   summary = ' '.join(summary.split())
     #   json_array[sent] = summary
     #   print "Abstract Sentence:  " + sent + " Top two sentences from body:  " + summary
    #print json.dumps(json_array)
    outputfile = open('sentences.json', 'wb')
    j = json.dumps(json_array, indent=4)
    print >> outputfile,j
    outputfile.close()
