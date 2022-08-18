﻿CS657A : INFORMATION RETRIEVAL
ASSIGNMENT 1

NOTE: The code is path sensitive i.e. don't change any path/directory and make proper folders as mentioned in the README, otherwise code might fail .
NOTE: While unzipping the files, please put these given folders[Answer_1, Answer_2, Answer_3, Answer_4, Answer_5] in main working directory with "english-corpora" folder.

Dependencies used:
python3
ntlk
pickle  
numpy
num2words 
math
sys 
os 
re
pandas


Folders and Files attached:
	1. Please follow this URL to download english-corpora.zip:
    •  https://drive.google.com/file/d/1KPhVys3Wkqz5Rrm3JmRD4HwVSr0mgaaK/view
      NOTE – Please download and extract above zip file and put “english-corpora” folder in current working directory.
      
   	2. 'Answer_1/' folder
		A) This folder will contain the files generated from question-1 python file after preprocessing the entire corpus
		B) In total 3 files will be generated
			B.1) 'post_dict' - this file contains the dictionary of lists, the keys are the unique words of the corpus and value is a list whose elements are the tuple with first value as document_id and second value as word frequency in the document.
			B.2) 'filed_Index' - this file contains the dictionary whose key is an integer and value is file_name, basically I made this to assign an integer index to a each file
			B.3) 'dict_norm' - this file contains the dictionary whose key is an integer index corresponding to a file and value is the norm of the document vector having tfidf values for each word of document

   	3. 'Answer_1.py' is the python file for question-1, running this file will take some time
		A) when this file is executed, it will generate three pickle files in 'Answer_1/' folder
		B) running this file is necessary to run other questions as files generated by it used by other models
	
	4. 'Answer_2/' folder
		A) 'Answer_2_Boolean_Retrieval_Model' - Boolean retrieval model
		B) 'Answer_2_TF_IDF.py' - TF IDF model
		C) 'Answer_2_BM_25.py' - BM-25 Model
	
	7. 'Answer_3/' folder 
		A) query_file.txt - a query file which contains [QueryId, Query] 
		B) ground_truth.txt - a file which contains [QueryId, Iteration, DocId, Relevance]

	8. 'Answer_4/' folder
		A) Boolean.txt
		B) TFIDF.txt
		C) BM_25.txt

	9. Makefile - a make file to run all models at once
	10. README.txt


Description:

There are different file structures of pickle files generated as shown below:
	file_norms : {idx : norm_value}
	postingLists : {‘word’ : {index : frequency of 'word'}}
	fileIndex : {index : (filename, #words in file)}

Commands to run: python3 Answer_1.py or python Answer_1.py
NOTE – Please execute Answer_1.py to generate above pickle files.

II. For Answer_2, there are 3 python files in which all 3 models have been implemented in respected python files. Each model takes “query_file.txt” from command line as an argument and returns “*.txt” corresponding to each model with comma separated format in the 'Question-4/' folder with respective model name. 

Commands in Makefile will look like these to run each model: 
	A) python3 ../Answer_2/Answer_2_Boolean_Retrieval_Model.py $(ARGS)
	B) python3 ../Answer_2/Answer_2_TF_IDF.py $(ARGS)
	C) python3 ../Answer_2/Answer_2_BM_25.py $(ARGS)
	
ARGS will accept the query file name from command line and pass it in each model .py file, each .py file of model will read it and open it and generated output in 'Answer_4/' directory

Command to run Makefile:
make run ARGS=”query_file_name.txt”


"SHORT SUMMARY :"

-> First Go the link of drive and download the corpus, save it as 'english-corpora/*' directory contain all documents
-> install dependencies  "num2words and nltk" using pip
-> when above step is done correctly, run the command "make cleaning", this command will run the first question and generate all necessary preprocessed files.
-> after that run the command " make run ARGS='query_file_name.txt' ", it will run all the models on the queries in query file and make three result documents in 'Question-4/' folder which contain top-5 documents for each query
-> Note that the file containing query must be present in Answer_3/ folder.