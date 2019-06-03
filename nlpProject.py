# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 09:40:41 2019

@author: simge
"""



import nltk
import pandas as pd
import string
import re
from nltk import ngrams
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.collocations import *
from nltk.corpus import brown

nltk.download('punkt')
nltk.download('stopwords')


#************************************* PHASE - 1 ******************************** 

stop_words = stopwords.words("ENGLISH")
ps=PorterStemmer()
ss=SnowballStemmer("english")

reviews = pd.read_csv("reviews.csv")
testData = reviews[:10]
testText = testData["Text"]


#tokenize the text into words
tokenized=[]
for txt in testData["Text"]:
    sentence = nltk.word_tokenize(txt)
    tokenized.append(sentence)
    

    
#take out the stop words of English language from the tokenized text
takeOutStopWords = []
for txt in tokenized:
    temp=[]
    for word in txt:
        if word.lower() not in stop_words:
            temp.append(word)
    takeOutStopWords.append(temp)

    
#apply stemming techniques to find the words’ roots 
#PorterStemmer    
PorterStemmer=[]
sentence=[]
for txt in takeOutStopWords:
    for word in txt:
        sentence.append(ps.stem(word))
    PorterStemmer.append(sentence)

#SnowballStemmer
SnowballStemmer=[]
sentence=[]
for txt in takeOutStopWords:
    for word in txt:
        sentence.append(ss.stem(word))
    SnowballStemmer.append(sentence)


#list of all words
words=[]
for txt in PorterStemmer:
    for word in txt:
        words.append(word)


#Display the frequency distribution information of the stemmed text.
freqDist=FreqDist(words)
print(freqDist)


#Display the most frequent 10 stems.
others = ['.',',','!','?','br','>','<','-','n','/']
temp=[]
for word in words:
    if word not in others:
        temp.append(word)

freqDistCommon10=FreqDist(temp)
freqDistCommon10.most_common(10)
        
    
#Visualize the frequency distribution using graphical plots
freqDist.plot()
    

#List all the words from the text which have more than 10 letters.
moreThan10Letters=[]
for word in words:
  if len(word)>10:
     if word not in moreThan10Letters:
         moreThan10Letters.append(word)
print(moreThan10Letters)








#************************************* PHASE - 2 ******************************** 




s ="Pamuk created an actual Museum of Innocence, consisting of everyday objects tied to the narrative, and housed them at an Istanbul house he purchased. Pamuk collaborated on a documentary""The Innocence Of Memories” that expanded on his Museum of Innocence. Pamuk stated that""(Museum of Dreams will) tell a different version of the love story set in Istanbul through objects andGrant Gees wonderful new film” Pamuk created an actual Museum of Innocence. Pamuk stated that ”(Museum of Dreams will) tell a diferent version of the love story set in Istanbul through objects and Grant Gees wonderful new film.”."

#a function named as preprocess;returns the tokenized version of the text that does not contain neither any stop words nor any punctuations
def preprocess(text):
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', s) 
    txt=[word for word in nltk.word_tokenize(text.lower()) if word not in stop_words]
    return txt;
    
    #punctuation=list(string.punctuation)
    #stop = list(string.punctuation)+stop_words
    #txt=[word for word in nltk.word_tokenize(text.lower()) if word not in stop]
    #return txt;
 

 preprocess(s)
 

#a function named as mostFrequent, which takes tokenized version of text and a number n as parameters, and returns the number of the occurrences of the frequent words.
def mostFrequent(text , n):
   txt = preprocess(text)
   fredist =FreqDist(txt)
   return fredist.most_common(n);
   
mostFrequent(s,3)


#a function named as displayNgrams,which takes tokenized text and a number n as parameters, and displays n grams only as many as the desired n
def displayNgrams(text,n):
    token = preprocess(text)
    return ngrams(token,n)

print(list(displayNgrams(s,3)))


# a function named as mostFreqBigram, which takes frequency of the bigram, number of the bigrams that are going to be listed and a list of bigrams, and returns only the bigrams with the given frequency rate
def mostFreqBigram(freq , n):***
    bgs=displayNgrams(s,n)
    fdist = nltk.FreqDist(bgs)
    l=list()
    for k,v in fdist.items():
        if v == freq:
            l.append(k)
    return l;
    
mostFreqBigram(3,2)


# a function, which takes bigrams as parameters, and returns the top 10 bigrams.
def probableOccur(bigrams):
    bigramMeasures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_documents(bigrams)
    return finder.nbest(bigramMeasures.pmi, 10)  # raw_freq 

print(probableOccur(displayNgrams(s, 2)))


# a function that returns the score information of the bigrams that are equal to or more frequent than 2
def scoreInformation(bigrams):
    bigramMeasures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_documents(bigrams)
    finder.apply_freq_filter(2)
    score = finder.score_ngrams(bigramMeasures.pmi)
    return score
    
scoreInformation(displayNgrams(s,2))


# a function that produces a list of words. Each word will have speech tag along with them
def listOfWords(text):
    tokenize = nltk.word_tokenize(text)
    return nltk.pos_tag(tokenize)

listOfWords(s)


# a function numOfTags that takes tagged list and returns only the most common tags
def numOfTags(taggedList):
    list = {}
    for x in taggedList:
        if x[1] in list:
            list[x[1]] +=1
        else:
            list[x[1]] = 1
    return list

numOfTags(listOfWords(s))


# a function, which takes two parameters (one of them for tagged text, and the other one is for a tag), displays the words in descending order for the speciﬁed tag
def specifiedTag(taggedText,tag):
    text = list()
    for k,v in taggedText:
        if tag in v:
            text.append(k)
    return text

specifiedTag(listOfWords(s),'NN')


#list all the words with number of occurrences information, frequency information and rank information along with their speech tags
def function(txt):
    tokenize = nltk.word_tokenize(txt)
    tokenize = [word.lower() for word in tokenize if word.isalpha()]
    sortedFreq =  sorted(((value,key) for (key,value) in FreqDist(tokenize).items()),reverse = True)

    
    
    freq=[]
    word=[]
    for i in sortedFreq:
        freq.append(i[0]*100/len(tokenize))
        word.append(i[1])
    
   
    index=[]
    for i in range(1,len(freq)+1):
        index.append(i)
        
        
    count=[]
    for i in sortedFreq:
         count.append(i[0])
    
    
    
    freqRank=[]
    for i in index:
        rs=i*freq[i-1]/100
        freqRank.append(rs)
        
    rank_df = pd.DataFrame(data = {'Rank': index})
    words_df = pd.DataFrame(data = {'Word': word})
    freq_df = pd.DataFrame(data = {'Frequency': freq})
    freqRank_df = pd.DataFrame(data = {'Freq X Rank': freqRank})
    counts_df = pd.DataFrame(data = {'Counts': count})
    df_final = pd.concat([rank_df,words_df,counts_df,freq_df,freqRank_df],axis = 1)
    
    return df_final
    

function(s)








#************************************* PHASE - 3 ******************************** 


# a function that constructs a lexicon that maps words of the English language to lexical categories.
def constructsLexicon():
    text = nltk.Text(word.lower() for word  in brown.words()[:1000])
    list={}
    for word, pos in sorted(nltk.pos_tag(text)):
        if word not in list:
            list[word] = pos

    return list

constructsLexicon()



# a function which constructs a grammar that defines the structure of a sentence.
grammar = nltk.CFG.fromstring("""
S -> NP VP
PP -> P NP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
Det -> 'an' | 'my'
N -> 'elephant' | 'pajamas'
V -> 'shot'
P -> 'in'
""")

def constructGrammar(sentence):
    parser = nltk.ChartParser(grammar)
    for tree in parser.parse(sentence):
      print(tree)
        
sentence = 'I', 'shot', 'an', 'elephant', 'in', 'my', 'pajamas'
constructGrammar(sentence)




#************************************* PHASE - 4 ********************************

#reading dataset from csv file

df = pd.DataFrame.from_csv("reviews.csv")

#taking the reviews column
reviews = df.iloc[:,8]
points = df.iloc[:,5]


review_Array = []
for i in reviews:
    review_Array.append(i)

def method(index):
    review = re.sub('[^a-zA-Z]',' ',index)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review

corpus = []
for i in review_Array:
    corpus.append(method(i))


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=10000)
X = cv.fit_transform(corpus).toarray()
y = points

from sklearn.cross_validation import train_test_split
X_train, X_test , y_train, y_test =train_test_split(X,y,test_size = 0.30,random_state = 0)


print ("GaussianNB")

from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(X_train,y_train)
y_pred_NB = classifier.predict(X_test)


print("Accuracy score for NaviveByaes")

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred_NB,y_test))
print(classification_report(y_test,y_pred_NB))



print("Linear Regression")

from sklearn.linear_model import LinearRegression

clf = LinearRegression(normalize=True)
clf.fit(X_train,y_train)
y_pred_linear = clf.predict(X_test)


print("r^2 score for Linear")
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred_linear))

print("SVM")
from sklearn.svm import SVC
model = SVC(kernel = 'linear')
model.fit(X_train, y_train)
y_pred_SVM = model.predict(X_test)


print("confusion matrix for svm")

print(confusion_matrix(y_test,y_pred_SVM))
print(accuracy_score(y_pred_SVM,y_test))
print(classification_report(y_test,y_pred_SVM))


print("KNN")
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_knn = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_knn)

print(accuracy_score(y_pred_knn,y_test))












    
            
