Q2.2
import json
with open('/Users/arrowlittle/Desktop/data/wiki_02.json', 'r') as f:
	data=json.load(f)


print(data[0]['url'])
print(data[0]['id'])
print(data[0]['title'])
print(data[0]['text'])

raw_docs = []
for i in range(100):
	raw_docs.append(data[i]['text'])


print repr(raw_docs[15]).decode("unicode-escape")


# Tokenizing text into bags of words

import nltk
from nltk.tokenize import word_tokenize
tokenized_docs = [word_tokenize(doc) for doc in raw_docs]

print repr(tokenized_docs[2]).decode("unicode-escape")


# Removing punctuation

import re
import string
regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html

tokenized_docs_no_punctuation = []

for review in tokenized_docs:
    new_review = []
    for token in review:
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            new_review.append(new_token)
    
    tokenized_docs_no_punctuation.append(new_review)
    

print repr(tokenized_docs_no_punctuation[2]).decode("unicode-escape")



# Cleaning text of stopwords

from nltk.corpus import stopwords

tokenized_docs_no_stopwords = []

for doc in tokenized_docs_no_punctuation:
    new_term_vector = []
    for word in doc:
        if not word in stopwords.words('english'):
            new_term_vector.append(word)
    tokenized_docs_no_stopwords.append(new_term_vector)



print repr(tokenized_docs_no_stopwords[0]).decode("unicode-escape")




#word stemming
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter = PorterStemmer()
snowball = SnowballStemmer('wiki')
wordnet = WordNetLemmatizer()

preprocessed_docs = []

for doc in tokenized_docs_no_stopwords:
    final_doc = []
    for word in doc:
        final_doc.append(porter.stem(word))
        #final_doc.append(snowball.stem(word))
        #final_doc.append(wordnet.lemmatize(word))
    
    preprocessed_docs.append(final_doc)


print repr(preprocessed_docs[2]).decode("unicode-escape")



#create sum format for data feature vector

pre_svm=[]
for i in range(4):
    string = ' '.join(preprocessed_docs[i])
    pre_svm.append(string)

print repr(pre_svm[2]).decode("unicode-escape")

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import dump_svmlight_file
import numpy as np 
import os

y = [0,0,0,0]
i = 0

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(pre_svm)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)
f = open('/Users/arrowlittle/Desktop/data/wiki_libsvm.txt', 'w')
dump_svmlight_file(tfidf, y, f, zero_based=False)
f.close()

print tfidf.toarray()


#split data

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

data = spark.read.format("libsvm") \
    .load("/Users/arrowlittle/Desktop/data/wiki_libsvm.txt")

splits = data.randomSplit([0.9, 0.1], 1234)
train = splits[0]
test = splits[1]

train.show(5)



#build model
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

model = nb.fit(train)

predictions = model.transform(test)
predictions.show()

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))
