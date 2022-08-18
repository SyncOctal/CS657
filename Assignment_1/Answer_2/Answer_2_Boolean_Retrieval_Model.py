#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing important basic modules that are needed to create IR system!
print("Importing libraries...")
import nltk
import os
import string
import math
import sys
import pickle
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


# query = sys.argv[1]


# In[1]:


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


#Boolean Retrieval Model Implementation
def BRM(query):
    t_diff_words = token_stem(query)
    diff_words = []
    operators=[]
    for i in range(1,len(diff_words)):
        operators.append("and")

    for item in t_diff_words:
        diff_words.append(porter.stem(item))
    
    all_list = []
    for item in diff_words:
        temp_list=[0]*len(filed_index)#Creating bitVector of size = #files for each query word
        if item in post_dict.keys():
            for x in post_dict[item]:
                temp_list[x[0]]=1
        all_list.append(temp_list)
    

    for i in operators:#Performing bitWise & between each bitVector
        v1 = all_list[0]
        v2 = all_list[1]
        temp = [x&y for x, y in zip(v1, v2)]
        all_list.pop(0)
        all_list.pop(0)
        all_list.insert(0, temp)
    temp_list=all_list[0]
    
    f_list = change_index(filed_index, temp_list)#Changing idx name to filename
    f_list.sort(reverse=True,key=lambda x:x[1])
    return f_list[:11]

def change_index(filed_index, temp_list):
    final_list = []
    for i,item in enumerate(temp_list):
        final_list.append((filed_index[i][0],item))
    return final_list

# In[ ]:

def query_run():
    query = str(sys.argv[1])#Argument passing
    q_file = pd.read_csv(sys.path[0]+'/../Answer_3/'+query, sep = '\t')
    for item in q_file.values:
        print('running Boolean Model for ',item[0])
        temp=BRM(item[1])[:5]#Result of boolean model stores in temp 
        # print(temp)
        q_id = item[0]
        d_id = list(map(lambda x:x[0], temp))
        rele = list(map(lambda x:x[1], temp))
        for i,j in enumerate(temp):
            res_file.loc[len(res_file.index)] = [q_id, 1, j[0], rele[i]]#storing each record rowWise in dataframe
    res_file.to_csv(sys.path[0] +'/../Answer_4/Boolean.txt', sep = ',', index=False)



#returns doc_rank which contains doc_names rankwise wrt query
# doc_rank = BRM(query)
# print(doc_rank)

#Creating DataFrame
res_file = pd.DataFrame(columns = ['QueryID', 'Iteration', 'DocID', 'Relevance'])
#Main function to call!
query_run()




