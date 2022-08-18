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

#all english-corpora docs in a single list(all_files)!
print('Reading all files from database...')
all_files = [] 
for i in os.listdir(sys.path[0]+'/../english-corpora/'):  
    all_files.append(i)
all_files.sort()
print("Done!")


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
#calculating average length of documents
count = 0   
for item in overall_words:
    count += len(overall_words[item])
L = count/len(all_files)

def normalizer(d_j):
    return len(filed_index[d_j])/L
#function to calculate term frequency
def TF(q_i, d_j):
    if q_i in post_dict.keys():
        idx_list = post_dict[q_i]
        for idx in idx_list:
            if idx[0] == d_j:
                return idx[1]        
    return 0
            
#function to calculate doc_frequency
def DF(q_i):
    if q_i in post_dict.keys():
        return len(post_dict[q_i])
    else:
        return 0
#function to calculate inverse doc frequency
def IDF(q_i):
    return numpy.log((len(all_files)-DF(q_i)+0.5)/(DF(q_i)+0.5))


# In[ ]:


def BM_25(query):
    t_diff_words = token_stem(query)
    diff_words = []
    for item in t_diff_words:
        diff_words.append(porter.stem(item))
    
    for d_j in filed_index:
        bm_result = 0
        for q_i in diff_words:
#             print(IDF(q_i), TF(q_i, d_j))
            bm_result += IDF(q_i)*((int(TF(q_i, d_j))*(k+1))/(int(TF(q_i, d_j))+k*(1-b+b*normalizer(d_j))))
        if bm_result:
            bm_dict.append((filed_index[d_j][0], bm_result))
    bm_dict.sort(key = lambda x : x[1], reverse = True)
    return bm_dict

def query_run():
    query = str(sys.argv[1])
    q_file = pd.read_csv(sys.path[0]+'/../Answer_3/'+query, sep = '\t')
    for item in q_file.values:
        print('running BM25 for ',item[0])
        temp=BM_25(item[1])[:5]
        q_id = item[0]
        d_id = list(map(lambda x:x[0], temp))
        rele = list(map(lambda x:x[1], temp))
        for i,j in enumerate(temp):
            res_file.loc[len(res_file.index)] = [q_id, 1, j[0], 0 if rele[i] == 0 else 1 ]
    res_file.to_csv(sys.path[0]+'/../Answer_4/BM_25.txt', sep = ',', index = False)

res_file = pd.DataFrame(columns = ['QueryID', 'Iteration', 'DocID', 'Relevance'])
#Tuned hyperparameters
k, b = 2, 0.75
bm_dict = []
#Main calling function!
query_run()






# In[ ]:




