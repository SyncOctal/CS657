#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing important basic modules that are needed to create IR system!
print("Importing libraries...")
import nltk
import os
import string
import math
import pickle
import sys
import time
import numpy
import pandas as pd
from numpy import linalg as LA
from num2words import num2words
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
print("Done!")



# In[ ]:

print("Reading data from pickle...")
with open(sys.path[0]+'/../Answer_1/overall_words.pickle', 'rb') as handle:
    overall_words = pickle.load(handle)
with open(sys.path[0]+'/../Answer_1/filed_index.pickle', 'rb') as handle:
    filed_index = pickle.load(handle)
with open(sys.path[0]+'/../Answer_1/post_dict.pickle', 'rb') as handle:
    post_dict = pickle.load(handle)
with open(sys.path[0]+'/../Answer_1/norm_dict.pickle', 'rb') as handle:
    norm_dict = pickle.load(handle)
print("Done!")

# In[ ]:


#all english-corpora docs in a single list(all_files)!
print('Getting all .txt files...')
all_files = [] 
for i in os.listdir(sys.path[0]+"/../english-corpora/"):  
    all_files.append(i)
all_files.sort()
print("Done!")

# In[ ]:


#english_stopwords is list of stopwords!
english_stopwords = stopwords.words('english')
#calling stemmer!
porter = PorterStemmer()

#removeStopWords helps to remove english stopwords!
def removeStopWords(tokens):
    return [t for t in tokens if t not in english_stopwords]

#token_stem helps in cleaning and processing the text and convert it to stemmed and tokenized tokens!
def token_stem(strr):
    strr = strr.lower()
    #removing \n, \t and \'s from .txt files!
    strr.replace('\n', ' ')
    strr.replace('\t', ' ')
    strr.replace('\'s', ' ')
    strr.replace("'",' ' )
    #removing punctuation from .txt files!
    file_content = strr.translate(str.maketrans('', '', string.punctuation))
    #removing ascii characters from .txt files!
    strencode = file_content.encode("ascii", "ignore")
    strdecode = strencode.decode()
    #tokenization!
    tokens = nltk.word_tokenize(strdecode)
    #converting digits to english words!
    num_tokens=numToWords(tokens)
    #removing stopwords!
    s_tokens = removeStopWords(num_tokens)
    
    return s_tokens
        
#removing digits from tokens!
def numToWords(tokens):
        for token in tokens:
            if token.isdigit() and len(token)<4:
                index = tokens.index(token)
                x = int(token)
                tokens[index] = num2words(x)
        final_str = ' '.join(tokens)
        final_tokens=nltk.word_tokenize(final_str)
        return final_tokens


# In[ ]:
#doc_dict contains idx --> doc_vector
doc_dict = {}

def TF_IDF(query):
    #query vector
    q_vector = []
    t_diff_words = token_stem(query)
    diff_words = []
    for item in t_diff_words:
        diff_words.append(porter.stem(item))
    
    for word in set(diff_words):
        tfidf = query_vector(word, diff_words)
        q_vector.append(tfidf)
        
    for fid in filed_index.keys():
        file_vec = []
        for word in set(diff_words):
            if word in post_dict.keys():
                tf = 0
                temp_list = post_dict[word]
                for item in temp_list:
                    if item[0] == fid:
                        tf = item[1]
                        break
                df = len(temp_list)
                idf = math.log((len(all_files)+1)/(df+1))
                tfidf = tf * idf
                file_vec.append(tfidf)
        doc_dict[fid] = file_vec
    return cosine_s(q_vector, doc_dict)
        
def cosine_s(q_vec, d_vec):
    #res_dict contains filename --> cosine score
    res_dict = []
    for vec in d_vec:
        d_norm = norm_dict[vec]
        result = numpy.dot(q_vec, d_vec[vec])/(LA.norm(q_vec)*d_norm)
        if result:
            res_dict.append((filed_index[vec][0], result))
    res_dict.sort(key = lambda x : x[1], reverse = True)
    return res_dict

def query_vector(word, diff_words):
    term_f = diff_words.count(word)
    inv_doc_f = math.log((len(all_files)+1)/(len(post_dict[word])+1))
    return term_f*inv_doc_f



def query_run():
    query = str(sys.argv[1])
    q_file = pd.read_csv(sys.path[0]+'/../Answer_3/'+query, sep = '\t')
    for item in q_file.values:
        print('running TF-IDF for ',item[0])
        temp=TF_IDF(item[1])[:5]
        # print(temp)
        q_id = item[0]
        d_id = list(map(lambda x:x[0], temp))
        rele = list(map(lambda x:x[1], temp))
        for i,j in enumerate(temp):
            res_file.loc[len(res_file.index)] = [q_id, 1, j[0], 0 if rele[i] == 0 else 1 ]
    res_file.to_csv(sys.path[0]+'/../Answer_4/TFIDF.txt', sep = ',', index = False)

res_file = pd.DataFrame(columns = ['QueryID', 'Iteration', 'DocID', 'Relevance'])
query_run()







