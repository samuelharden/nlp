import re
import random
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
import os
import json
from gensim import corpora

class MyCorpus(corpora.TextCorpus):
    def get_texts(self):
        files = os.listdir('../processed_papers_t/')
        text = ''
        json_data={}
        for fl in files:
            print fl
            json_data = json.load(open('../processed_papers_t/'+fl))
            for  val in json_data["abstract_sentences"].values():
                text += " "
		if val is not None:
		    text += val
            for val in json_data['body_sentences'].values():
                text += " "
		if val is not None:
		    text += val
	    #print text
	    yield self.preprocess_text(text)

stop = stopwords.words('english')
add_stopwords = ['said', 'mln', 'billion', 'million', 'pct', 'would', 'inc', 'company', 'corp']
stop += add_stopwords

def ie_preprocess(document):
    document = re.sub('[^A-Za-z ]+', '', document)
    document = ' '.join([i for i in document.lower().split()
                        if i not in stop])
    document = nltk.word_tokenize(document)
    return document

def remove_infrequent_words(docs):

    """Remove all the words that only occur once"""

    from collections import defaultdict
    frequency = defaultdict(int)
    for doc in docs:
        for token in doc:
            frequency[token] += 1

    docs = [[token for token in doc if frequency[token] > 1]
             for doc in docs]
    return docs

def run():

    """Import the Reuters Corpus which contains 10,788 news articles"""

    from nltk.corpus import reuters
    raw_docs = [reuters.raw(fileid) for fileid in reuters.fileids()]

    # Select 100 documents randomly
    rand_idx = random.sample(range(len(raw_docs)), 100)
    raw_docs = [raw_docs[i] for i in rand_idx]

    # Preprocess Documents
    tokenized_docs = [ie_preprocess(doc) for doc in raw_docs]

    # Remove single occurance words
    docs = remove_infrequent_words(tokenized_docs)

    # Create dictionary and corpus
    #corpus = [dictionary.doc2bow(doc) for doc in docs]
    corpus = MyCorpus()
    print corpus.get_texts()
    dictionary = corpora.Dictionary()
    dictionary.add_documents(corpus.get_texts())
    #corpus.init_dictionary(dictionary)
    # Build LDA model
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=100, update_every=0, chunksize=3000, passes=20)
    for topic in lda.show_topics():
        print topic


if __name__ == '__main__':
    run()
