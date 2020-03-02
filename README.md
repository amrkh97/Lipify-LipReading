Lipify - A Lip Reading Application
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
**NOTE: Speaker 21 is missing from the GRID Corpus dataset due to technical issues.

---

##### Datset Segmentation Steps:
1. Run DatasetSegmentation.py
2. Run Pre-Processing/frameManipulator.py
<br>
* After running the above files, all result videos will have 30 FPS and 1 second long.

---

##### TODOs:

* ~~Dataset preprocessing module~~
* ~~Initial Neural networks' architecture~~
* Facial detection algorithm
* Feature extraction module
* Proper documentation for the whole project
* Optimization of the networks' architectures

---

#### License:
[MIT License](https://github.com/amrkh97/Lipify-LipReading/blob/master/LICENSE.md)
