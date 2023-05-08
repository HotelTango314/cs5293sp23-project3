# PROJECT 3 README
## Henry Thomas -- CS5293sp23-Project3 FINAL PROJECT

### 1. HOW TO INSTALL

The project can be installed by running the following code:

```
pipenv install project3
```
---

### 2. HOW TO RUN

TO RUN FROM THE ROOT DIRECTORY USE THE FOLLOWING 

```
pipenv run python project3.py --document testcity.pdf
```

--document is the only implemented flag. A sample pdf file is saved in the root directory for use in testing functionality of the file (testcity.pdf)

Visit this link for a video example of how to run:
[youtube.com](https://youtu.be/wMRusK8uNlE)

### 3. DESIGN CONSIDERATIONS

project3.py is the name of the source file that extracts the text data from the provided new pdf file which then vectorizes and asssigns it to a cluster based on two models saved using pickle to the directory:
vectorizer.pkl, and model.pkl.

Vectorizer.pkl is a pipeline object that was trained on all the PDF documents provided and runs this data through tfidfvectorizer, TruncatedSVD, and Normalizer as recommended in the sklearn documentation and as I used it in the project 2 clustering assignment and vectorizes the data. This pipeline object was saved with pickle so that future documents that we would like to be assigned to a cluster could be vectorized the same way as the originals.

The second pickle model is the model.pkl file which is the AgglomerativeClustering SKLEARN model with K=2. We use this to actually cluster the new documents.

I spent a lot of time trying to find the best k, and trying to figure out which words to exclude. My main strategy in excluding the words was to use a tfidfvectorizer object to identify words that appear in 97.5% or more of the documents. If a word appears in this many of the documents it is likely one of those words like 'smart' or 'city' which doesn't help much to discern between a pile of documents about smart cities. Since TfidfVectorizers can automatically remove words that appear in more than X% of the documents, as well as words that appear in less than X% of the documents (helpful for eliminating words that are highly specific to one document, but don't help with understanding meaning, i.e. the name of a particular city). I was able to do this very easily with just the arguments of the tfidfvectorizer constructor. That said, after doing that (and also adding a method to the text cleaner that removed words not in the medium sized spacy english model) there were still a few words appearing which were not words i.e. 'st' and 'zp' which I went back and made a special list for to remove. A lot of time was spent looking for that perfect value of X at the top of the document frequency value to cut out frequently occuring words by document.

After doing all this I was able to run the model and get a pretty high silhouette score. The one thing I was not too happy with was the imbalance of the cluster labeling. If K=2 I was hoping for more of a 50/50 split between the two clusters, but it was more of a 15/85 split.


### 4. TESTS

Tests were implemented for each grouping of models to verify operability.

### 5. OTHER NOTES

This is the final - so I'd like to say that I really enjoyed this class. I felt like it gave me a lot to think about and really laid the groundwork for continued learning in this area. I haven't had as much exposure as I have had to sklearn as I have in this class and it is a really great tool and so much excellent documentation which I look forward to continuing to explore as I further my career in this field.

Note to self for future use:
COMMAND TO RUN jupyter notebook:
pipenv run jupyter-lab --no-browser --ip 0.0.0.0 --ServerApp.iopub\_data\_rate\_limit=1.0e10
