#Pre-req libraries
import re
import nltk
from nltk.corpus import stopwords
import nltk
import heapq
import imp

#Files to read and write to
circumstances_raw = open('circumstances.txt','r')
circumstance_summary = open('cirucmstances_final.txt','w')

#Pre-processing
context = circumstances_raw.read() #Contains raw article
formatted_context = re.sub('[^a-zA-Z]',' ',context) #contains formatted article for tf-idf
formatted_context = re.sub(r'\s+',' ',formatted_context)

#Tokenizing
sentence_list = nltk.sent_tokenize(context)

#Defining and removing stopwords
stopwords = nltk.corpus.stopwords.words('english')

#Finding weighted frequency of occurences
word_frequencies = {}
for word in nltk.word_tokenize(formatted_context):
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

max_frequency = max(word_frequencies.values()) #Max word freq

#tf-idf
for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word]/max_frequency)

#Sentence Score calculations
sentence_scores = {}

for sent in sentence_list:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' '))< 30:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

#Using Heap Queue, retreiving the top sentences and writing them to a file
summary_sentences= heapq.nlargest(25, sentence_scores, key=sentence_scores.get)
summary = ' '.join(summary_sentences)
circumstance_summary.write(summary)

circumstances_raw.close()
circumstance_summary.close()
