# Lipify - A Lip Reading Application
---

##### Project Dependencies:


* Python>=3.7.1
* tensorflow>=2.1.0
* opencv-python>=4.2.0
* dlib
* moviepy>=1.0.1
* numpy>=1.18.1
* Pillow
* matplotlib
* tqdm
* pyDot
* seaborn
* scikit-learn
* imutils>=0.5.3


Note: All Dependencies can be found inside 'setup.py'

---

##### Project's Dataset Structure:

* GP DataSet/ <br> | --> align/ <br> | --> video/ <br>
* Videos-After-Extraction/ <br> | --> S1/ <br> | --> .... <br> | --> S20/
* New-DataSet-Videos/ <br> | --> S1/ <br> | --> .... <br> | --> S20/
* S1/ <br> | --> Adverb/ <br> |
--> Alphabet/ <br> |
--> Colors/ <br> |
--> Commands/ <br> |
--> Numbers/ <br> |
--> Prepositions/
<br>

---
##### Dataset Info:

We use the GRID Corpus dataset which is publicly available at this [link](http://spandh.dcs.shef.ac.uk/gridcorpus/)
<br>You can download the dataset using our script: GridCorpus-Downloader.sh
<br> which was adapted from the code provided [here](https://gist.github.com/KarthikMAM/d8ebde4db84a72b083df0e14242edb1a)
<br> <br>
To Download please run the following line of code in your terminal:
<br>`bash GridCorpus-Downloader.sh FirstSpeaker SecondSpeaker`
<br> where FirstSpeaker and SecondSpeaker are integers for the number of speakers to download
<br>
* NOTE: Speaker 21 is missing from the GRID Corpus dataset due to technical issues.

---

##### Datset Segmentation Steps:
1. Run DatasetSegmentation.py
2. Run Pre-Processing/frameManipulator.py
<br>
* After running the above files, all resultant videos will have 30 FPS and 1 second long.


---
##### CNN Models Training Steps:
* Model codes can be found in the directory "NN-Models"

* First you will need to change the common path
 value to the directory of your training and test data.

* Run Each network to start training.
* Early stopping was used to help stop
 the training of the model at its optimum validation accuracy.
 <br>
* Resultant accuracies after training on the data can be found in:
[Project Accuracies](https://github.com/amrkh97/Lipify-LipReading/blob/master/Project_Insights/Project_Accuracy.csv)

---
##### TODOs:

* ~~Dataset preprocessing module~~
* ~~Initial Neural networks' architecture~~
* ~~Facial detection algorithm~~
* ~~Optimization of the networks' architectures~~
* ~~Unittesting of project files~~
* Proper documentation for the whole project
---

#### License:
[MIT License](https://github.com/amrkh97/Lipify-LipReading/blob/master/LICENSE.md)
