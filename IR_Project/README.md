PREREQUISITES:

1. Python
	Libraries:
		1. Selenium
		2. Tesseract
		3. stanza
		4. translators
		5. spacy
		6. networkx
		7. matplotlib
		8. pandas
		9. tqdm
		10. nltk
		11. pickle
		12. inltk

2. Jupyter Notebook:
Jupyter notebook is a very popular and flexible development environment which lets us write and execute python code, display the output and any kind 
of visualization or plot, etc. in the same document. however, it is always advisable to install Python with Anaconda environment on your system as well.

3. Ramayana:
Basic information about Ramayana will be helpful. We have included some information in report.


STEPS TO FOLLOW BEFORE START:

This project consists of 5 python notebook and required data.

The execution order of the files is as follows:
	1. WebScrapping.ipynb
	2. DataPreprocessing.ipynb
	3. TenseClassification.ipynb
	4. BM25.ipynb
	5. KnowledgeGraph.ipynb


DESCRIPTION:

The project Zip has two folders:
	1. Codes
	2. Datasets

Codes:

1. WebScrapping.ipynb

	Introduction:
		This notebook connects to "https://www.wisdomlib.org/hinduism/book/the-ramayana-of-valmiki" and gets the ramayana english data by webscrapping using selenium webdriver into Datasets/RamayanaEng.json and Datasets/RamayanaEngSen.json in two different formats and using this scrapped data, creates the hindi translated ramayana data, storing it in Datasets/RamayanaHin.json.


2. DataPreprocessing.ipynb

	Introduction:
		This notebook takes the images of hindi ramayana pdf data, preprocesses and cleans it and gives the Datasets/RamayanaHi.json hindi dataset. It saves an intermediate result at Datatsets/ramayana.txt. The first cell performs text extraction from images (which takes time). The size of the images is around 1.5 gb. So, we uploaded the image_dataset folder on google drive here: https://drive.google.com/drive/folders/1sF6xhVP4l95vGNLb6L3UmQ7DAu1IE_AU?usp=sharing. To re-run the first cell, please download the google folder and put it in Datasets folder. The images are extracted from the following pdf: https://drive.google.com/file/d/14Jl288loWkURMUJGKlG9sphMGsHU7O9p/view?usp=sharing. 


3. TenseClassification.ipynb

	Introduction:
		This notebook takes the RamayanaHi.json, the pdf extracted dataset and Datasets/RamayanaHin.json, the scrapped translated dataset and gives 3 files for both stored in Datasets/HiFrompdf and Datasets/HiTranslated folders respectively. The 3 files correspond to past, present and future files for both.


4. BM25.ipynb

	Introduction:
		This notebook performs BM25 scored retrieval on the events classified as past in previous file. These past events from current chapter are matched with all events from previous chapters and top 5 relevant chapters are stored. The output is present in Datasets/EventsMatched.json


5.  KnowledgeGraph.ipynb

	Introduction:
		This notebook creates a knowledge graph based on the RamayanaEngSen.json dataset. 

You can refer to the notebooks for more details about each section


Datasets:

1. ramayana.txt
2. RamayanaEng.json
3. RamayanaEngSen.json
4. RamayanaHi.json
5. RamayanaHin.json

Outputs/Mid-results:

1. Chapters/*.txt
2. HinFrompdf/*.json
3. HinTranslated/*.json
4. EventsMatched.json

Project team:

1. Mandar Dhake(21111405,mkdhake21@iitk.ac.in)
2. Deepak Hegde(21211402,dkumar21@iitk.ac.in)
3. Ashutosh Patel(21111018,ashutoshp21@iitk.ac.in)
4. Harish Patidar(21111029,hpatidar21@iitk.ac.in)