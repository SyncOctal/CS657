#!/usr/bin/env python
# coding: utf-8

#Answer_1

# In[1]:
#Importing important basic modules that are needed to create IR system!
print("Importing and downloading libraries...")
import nltk
nltk.download('punkt')
nltk.download('stopwords')
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
print('Done!')

# In[2]:
#all english-corpora docs in a single list(all_files)!
all_files = [] 
for i in os.listdir(sys.path[0]+ '/../english-corpora/'):  
    all_files.append(i)
all_files.sort()


# In[3]:

#english_stopwords is list of stopwords!
english_stopwords = stopwords.words('english')
#calling stemmer!
porter = PorterStemmer()


# In[4]:

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

# In[5]:

print("Creating syntax of norm_dict, post_dict, unique_words, filed_index, overall_words structures...")
#Creating the structures to store processed data!
norm_dict = {}
#{idx --> float}

post_dict = {}
#{word --> [(idx, all_words_freq)]}

unique_words = set()
#set(unique_words)

filed_index = {}
#{idx --> (filename, total_words)}

overall_words={}
#{filename --> [all_tokens]}


# In[6]:


start_time=time.time()
idx = 0
print("storing data in filed_index, unique_words, overall_words...")
for file in all_files:
    tokens = token_stem(open(sys.path[0]+ '/../english-corpora/'+file).read())
    #stemming!
    stemmed_tok = []
    for token in tokens:
        stemmed_tok.append(porter.stem(token))
    #creating overall_words file!
    overall_words[idx]=stemmed_tok
    #creating unique words!
    temp_unique=set(stemmed_tok)
    unique_words.update(temp_unique)
    #creating index file!
    filed_index[idx]=(file,len(tokens))
    idx+=1
print("Data has been stored successfully in filed_index, unique_words, overall_words...")  
print("It took ",time.time()-start_time," seconds to store data in filed_index, unique_words and overall_words")


# In[7]:


#Creating posting dictionary
start_time=time.time()
print("Creating posting dictionary...")
for word in unique_words:
    post_dict[word]=[]

for idx in overall_words.keys():
    words=overall_words[idx]
    for w in set(words):
        post_dict[w].append((idx,words.count(w)))

print("posting dictionary has been created successfully")   
print("It took ",time.time()-start_time," seconds to create posting dictionary")


# In[8]:
#Creating norm dictionary
print("Storing data in norm dictionary...")
start = time.time()
for idx in overall_words.keys():
    words = overall_words[idx]
    tf_idf = []
    for w in set(words):
        term_freq = words.count(w)
        doc_freq = len(post_dict[w])
        idf = math.log((len(all_files)+1)/doc_freq)
        tf_idf.append(term_freq*idf)
    norm_dict[idx] = LA.norm(tf_idf)
print("data has been stored successfully in norm dictionary")
print("It took ",time.time()-start," seconds to create norm dictionary")


# In[9]:

#saving processed structured data in pickle format!
print("dumping structured data in pickle format...")
with open(sys.path[0]+'/../Answer_1/post_dict.pickle', 'wb') as f1:
    pickle.dump(post_dict, f1, protocol=pickle.HIGHEST_PROTOCOL)
with open(sys.path[0]+'/../Answer_1/filed_index.pickle', 'wb') as f2:
    pickle.dump(filed_index, f2, protocol=pickle.HIGHEST_PROTOCOL)
with open(sys.path[0]+'/../Answer_1/norm_dict.pickle', 'wb') as f3:
    pickle.dump(norm_dict, f3, protocol=pickle.HIGHEST_PROTOCOL)
with open(sys.path[0]+'/../Answer_1/overall_words.pickle', 'wb') as f4:
    pickle.dump(overall_words, f4, protocol=pickle.HIGHEST_PROTOCOL)
print("dumping has been done successfully")