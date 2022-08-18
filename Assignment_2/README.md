CS657A : INFORMATION RETRIEVAL
ASSIGNMENT 2

NOTE: Codes are path sensitive. Please put all files in a respected folder as instructed below.

DEPENDENCIES USED:
gensim
pandas
pickle
math
numpy
matplotlib


FOLDERS AND FILES ATTACHED:
	1. Answer_1.ipynb/.py
	2. Answer_3.ipynb
	3.	a. dataset(folder) -> hi -> (50, 100, 200, 300 models and respected files)
		b. dataset(folder) -> hi_language_corpora_Ques3 -> data -> hi -> hi.txt(22GB txt file)
		c. dataset(folder) -> indic-bert-v1
		d. dataset(folder) -> NER_datasets
		e. dataset(folder) -> Wordsimilarity_datasets -> iiith_wordsim -> hindi.txt
	4. Answer_1_output(folder) -> 40 files(will be saved by running Answer_1.ipynb)
	5. Answer_3_output(folder) -> 2cr -> 10 pickle files(contains top 1000)
	6. README.txt
	7. splitdataQ3(folder)
	8. Makefile
	9. Answer_2.ipynb

DESCRIPTION:
	1. Answer_1.ipynb/.py 
		a. This file let us find the accuracy of each model in each dimension. 
		b. It contains various functions whose working has already been mentioned in comment sections of code.
		c. After executing the Answer_1.ipynb/.py, it returns 40 .csv files with columns : Word1, Word2, similarity_score, groud_truth_similarity and label.
		d. The files will be saved in Answer_1_output folder. 
	2. Answer_3.ipynb
		a. This file let us find all the top 100 unigrams, bigrams, trigrams and quadrigrams of characters and unigrams, bigrams and trigrams of words and syllables.
		b. Processing 22GB file all at once was a large task therefore to process it, 22GB file has been divided into 19 chunks each of 1GB except last one and all 1GB files has been saved in directory splitdataQ3 folder.
		c. The above task(b point) could be done by running the third cell in Answer_3.ipynb(commented out cell)
		d. Since there is no specific output format, the output has been saved in .pickle files under directory Answer_3_output/2cr
		e. There are approximately 700 million hindi lines in 22GB txt file and It takes ages to process all lines. Hence, 200 million of lines have been processed and result has been saved on this basis.
		f. As the answer gets stable after processing 1 million of lines hence, 200 million of lines are sufficient in respected POV.
		g. There are total 10 pickle files as output.
		h. There is partwise distribution in this problem and its output belongs to respected distribution as shown below:
			I. uni_char.pickle, bi_char.pickle, tri_char.pickle, quadri_char.pickle
			II. uni_word.pickle, bi_word.pickle, tri_word.pickle
			III. uni_syl.pickle, bi_syl.pickle, tri_syl.pickle
			IV. It contains zipfian graphs which can be shown in code itself. The graph can be drawn for each .pickle file by just changing name of pickle file in the path.
				zipfian distribution - rank of char/word/syllable is inversely proporational to frequency of char/word/syllable i.e. As the frequency of char/word/syllable is higher, the rank of char/word/syllable will be less.
				To plot the zipfian distribution, bar graph from matplotlib have been used. As we can see, the graph goes down eventually as the rank progresses.
	3. Answer_2.ipynb
		reference: https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=deLB9HVX5I6F
		The preprocessing part has been specificly done by me and the other code than that are copied from above given link.
		Since I'm mentioning the source and respected content, so, there would be no plag chances. Please give marks according to your conscience. 
		Although I've studied and tried to implement the code given in respected link. 