import nltk
from pyspark import SparkConf, SparkContext
from nltk.corpus import stopwords
from operator import add
import sys
import re
from math import log

## Constants
APP_NAME = " HelloWorld of Big Data"
##OTHER FUNCTIONS/CLASSES

def removePunctuation(text):
    text=text.lower().strip()
    text=re.sub('[^a-zA-Z ]',' ', text)
    return text

def extract(topWords, append, art):
   features_list = []
   words = art.lower().split(" ")
   for i in topWords:
      denom = (float(words.count(i[0].lower()))/len(words))
      if denom == 0:
         features_list.append(0.0)
      else:
         features_list.append((denom))
   return [[append] + features_list]

def main(sc):
   stop_words = list(set(stopwords.words('english'))) + [""]
   labels = ['technology','business/Shefali', 'sports/Shefali','politics/Shefali']
   textRDD = []
   words = []
   wordcount = []
   wordcountSorted = []
   i = 0
   for label in labels:
      textRDD.append(sc.textFile(label).map(lambda x: removePunctuation(x)))
      words.append(textRDD[i].flatMap(lambda x: [i for i in x.split(' ') if i not in stop_words]).map(lambda x: (x, 1)))
      wordcount.append(words[i].reduceByKey(add))
      wordcountSorted.append(wordcount[i].sortBy(lambda a: a[1], ascending = False).top(50, key = lambda x: x[1]))
      i += 1
   
   # technology - 0
   # business - 1
   # sports - 2
   # politics - 3
   feature_list_new = []
   
   allLabel = list(set(wordcountSorted[0] + wordcountSorted[1] + wordcountSorted[2] + wordcountSorted[3]))
   feature_list_new.append(textRDD[0].flatMap(lambda art: extract(allLabel, 0, art)))
   feature_list_new.append(textRDD[1].flatMap(lambda art: extract(allLabel, 1, art)))
   feature_list_new.append(textRDD[2].flatMap(lambda art: extract(allLabel, 2, art)))
   feature_list_new.append(textRDD[3].flatMap(lambda art: extract(allLabel, 3, art)))
   
   features_ = feature_list_new[0].union(feature_list_new[1])
   features_ = features_.union(feature_list_new[2])
   features_ = features_.union(feature_list_new[3]).collect()
   
   thefile = open('output.txt', 'w')
   for item in features_:
      print>>thefile, item
   
   # Unkown Data for testing
   
   labels = ['unknowData/technology','unknowData/business', 'unknowData/sports','unknowData/politics']
   i = 0
   for label in labels:
      textRDD.append(sc.textFile(label).map(lambda x: removePunctuation(x)))
      words.append(textRDD[i].flatMap(lambda x: [i for i in x.split(' ') if i not in stop_words]).map(lambda x: (x, 1)))
      wordcount.append(words[i].reduceByKey(add))
      wordcountSorted.append(wordcount[i].sortBy(lambda a: a[1], ascending = False).top(50, key = lambda x: x[1]))
      i += 1
   
   # technology - 0
   # business - 1
   # sports - 2
   # politics - 3
   feature_list_new = []
   
   allLabel = list(set(wordcountSorted[0] + wordcountSorted[1] + wordcountSorted[2] + wordcountSorted[3]))
   feature_list_new.append(textRDD[0].flatMap(lambda art: extract(allLabel, 0, art)))
   feature_list_new.append(textRDD[1].flatMap(lambda art: extract(allLabel, 1, art)))
   feature_list_new.append(textRDD[2].flatMap(lambda art: extract(allLabel, 2, art)))
   feature_list_new.append(textRDD[3].flatMap(lambda art: extract(allLabel, 3, art)))
   
   features_ = feature_list_new[0].union(feature_list_new[1])
   features_ = features_.union(feature_list_new[2])
   features_ = features_.union(feature_list_new[3]).collect()
   
   thefile = open('outputUnknown.txt', 'w')
   for item in features_:
      print>>thefile, item
   

if __name__ == "__main__":

   # Configure Spark
   conf = SparkConf().setAppName(APP_NAME)
   conf = conf.setMaster("local[*]")
   sc   = SparkContext(conf=conf)
   # Execute Main functionality
   main(sc)

