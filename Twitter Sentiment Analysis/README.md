# Introduction
Twitter is one of the most popular platforms on which users can publish their thoughts and opinions, then we can use sentiment analysis to identify user experience or feelings towards a product
or idea, which benefits the companies to enhance their performance. So what is Sentiment Analysis, it is a NLP technique used to determine whether data is positive, negative or neutral. Sentiment analysis in Twitter tracks the problem of analyzing the tweets of the opinions the people express. The most common type of Sentiment Analysis is “Polarity Detections” which involves classifying statements as positive, negative and neutral. So with the use of Sentiment Analysis we can know the opinion of the crowd. And this is what I am going to perform this using Machine learning and NLP techniques.


# Data Description
Our dataset has 74681 rows and 4 columns (“TweetID”, “Entity”, “Sentiment”, “TweetContent”) for training data. And 1000 rows with the same columns for validation data.
Here is a sample from our dataset

![Sample Image](https://github.com/mostafasamii/Natural-Language-Processing-Projects/tree/main/Twitter%20Sentiment%20Analysis/repo_imgs/sample.png)


# Baseline Experiments

The goal is to design a model for Sentiment Analysis to classify the tweet whether it is Positive,
Negative or Neutral. In order to do that I performed the following this pipeline:
* Import Necessary Dependencies
* Read and Load the Dataset
* Exploratory Data Analysis (EDA)
* Word2Vec / BOW
* Preparing data for modeling
* Modeling
* Ensemble techniques
* Model Evaluation

In the phase of reading data I faced the problem of encoding problems, and the encoding method I
used to read the data was “ISO-8859-1” , but “utf-8” didn’t work well.

In the phase of EDA we have the following characteristics:
Training data statistics:

![Sample Image](https://github.com/mostafasamii/Natural-Language-Processing-Projects/tree/main/Twitter%20Sentiment%20Analysis/repo_imgs/datastatistics.png)

Target Distribution represented in 3 target classes, 0 for “Positive” Tweets, 1 for “Negative” tweets, and 2 for “Neutral” Tweets. I can notice that there is a fair feature distribution in the target classes.

![Sample Image](https://github.com/mostafasamii/Natural-Language-Processing-Projects/tree/main/Twitter%20Sentiment%20Analysis/repo_imgs/targetdist.png)

Distribution of tweets lengths: We can notice that most of the length is around (0.75~1.25) and we
have an approximate normally distributed features here

![Sample Image](https://github.com/mostafasamii/Natural-Language-Processing-Projects/tree/main/Twitter%20Sentiment%20Analysis/repo_imgs/tweetslengths.png)

Most frequent words repeated in the entire dataset:

![Sample Image](https://github.com/mostafasamii/Natural-Language-Processing-Projects/tree/main/Twitter%20Sentiment%20Analysis/repo_imgs/freqwords.png)

Word cloud is an approach to visualize words and counts of words from reviews columns of the dataset, which artistically depict the words at sizes proportional to their counts.

![Sample Image](https://github.com/mostafasamii/Natural-Language-Processing-Projects/tree/main/Twitter%20Sentiment%20Analysis/repo_imgs/wordcloud.png)

In the phase of applying Word2Vec and BOW:
* Word2Vec model: The usage of Word2Vec model was for the purpose of detecting synonymous words or suggesting additional words for a word, as the Word2Vec algorithm uses a NN model to learn associations from a large corpus of text. Word2Vec represents each distinct word with a particular list of numbers called a vector. The Vectors are chosen carefully such that a simple mathematical function (the cosine similarity between vectors) indicates the level of semantic similarity between the words, and I can see that here in the given example from the data for texting the word “Twitter” I got the following:

![Sample Image](https://github.com/mostafasamii/Natural-Language-Processing-Projects/tree/main/Twitter%20Sentiment%20Analysis/repo_imgs/word2vec_res.png)

We can see them in descending order, the word “account.My” is the closest word to “Twitter”

* Bag of Words (BOW): I have used the PorterStemmer while building the training the testing data. The PorterStemmer removes the suffixes to find the stem, it divides a word in regions and then
replaces or removes certain suffixes, if they are completely contained in said region. And I joined back the data with space to build the train/test corpus. The BOW model then simplifies the representation of the words as in the model, a text is presented as the bag of its words, and it generates the features which will be fed into the upcoming models.


In the phase of preparing data for modeling: I used K-fold Cross Validation to enhance the model accuracy as it is a resampling procedure used to evaluate the models I am going to use on our data sample. I have used k=10 as mentioned in the project descriptions, K refers to the number of groups that a given data sample is to be split into. As such the procedure is often called K-fold Cross Validation. Then I have applied StandardScaler to the data to scale it in the range between 1st Quartile and 3rd Quartile in order to get more robust data and remove any potential outliers as the median and the interquartile range provide better results and outperform the sample mean and variance.

In the phase of modeling:

![Sample Image](https://github.com/mostafasamii/Natural-Language-Processing-Projects/tree/main/Twitter%20Sentiment%20Analysis/repo_imgs/modelsres.png)

After training our models I got these results, as we can see the RandomForest Model here in this problem outperforms the remaining models, with Validation Acc of 0.91.

In the phase of Model Evaluation For the model evaluation of Voting Classifier I used the Macro Averaging method for the evaluations, as it computes the metric independently for each class and then takes the average hence all classes equally. We should only use it when we care about the overall performance of the classes, that’s why I needed it here as I care about the 3 classes (“Positive”, “Negative”, “Neutral”) to have predictions as close as possible


# Other Experiments

I used Ensembling techniques In order to enhance the model performance, so I applied the Ensembling techniques using VotingClassifier, with voting hard and weight of [2, 1] which means more weight for RandomForest, and this method is a voting method assigns various weights to the classifiers based on specific criteria an takes the vote between the LGBM and Random Forest model, when I did that we have Validation Score of 0.91465 which is better than the Score of the Random Forest alone.


# Future Work

* Hyperparameter tuning to optimize models accuracy  


# Overall Conclusion

Machine learning techniques perform reasonably well for classifying sentiment in tweets. Random Forest outstands the rest of the models but using Ensemble techniques improved the accuracy as the
single classifier (Random Forest) may not be the best approach. It would be interesting to see what the results are for combining different classifiers.


# Tools

* Google Colab Notebooks
* Python 2.7.12
* Google Drive for storing data


# External Resources

* https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model
* https://towardsdatascience.com/step-by-step-twitter-sentiment-analysis-in-python-d6f650ade58d
* https://www.analyticsvidhya.com/blog/2021/06/twitter-sentiment-analysis-a-nlp-use-case-for-beginners/
* https://monkeylearn.com/blog/sentiment-analysis-of-twitter/
* https://en.wikipedia.org/wiki/Sentiment_analysis
* https://aclanthology.org/W11-0705.pdf
* https://ieeexplore.ieee.org/abstract/document/6726818
* https://ojs.aaai.org/index.php/ICWSM/article/view/14185/14034
* https://github.com/sharmaroshan/Twitter-Sentiment-Analysis/blob/master/Twitter_Sentiment.ipynb
* https://www-nlp.stanford.edu/courses/cs224n/2009/fp/3.pdf
* http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.2031&rep=rep1&type=pdf
