# SG_DAT_Hern
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 22:09:54 2017

@author: hernhernchua
"""
#from __future__ import absolute_import
#from __future__ import division, print_function, unicode_literals
"exploring wuzzuf egyptian jobs portal data"


#basic packages

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load data to pandas frames
apps = pd.read_csv("Wuzzuf_Applications_Sample.csv") #applications data
postings = pd.read_csv("Wuzzuf_Job_Posts_Sample.csv") #job postings

#postings all unique ID

"rename columns of data"
postings.columns = ['jobID', 'city', 'jobTitle', 'jobCat1', 'jobCat2', 'jobCat3',
                    'jobInd1', 'jobInd2', 'jobInd3', 'minSalary', 'maxSalary',
                    'vacancies', 'careerLevel', 'yExp', 'postDate', 'postViews',
                    'JD', 'jobReq', 'payPeriod', 'currency']

apps.columns = ['appID', 'userID', 'jobID', 'app_date']

'''count total apps per job ID and merge into mastersheet'''

totalApp_perjob = apps.groupby('jobID')['userID'].nunique()
totalApps = pd.DataFrame({'jobID': totalApp_perjob.index, 'totalApp': totalApp_perjob.values})

#merge apps into postings
mastersheet = pd.merge(postings, totalApps, on = 'jobID')

#mastercopy = mastersheet.sort_values(by='totalApp', ascending=1)
mastercopy = mastersheet[mastersheet['JD'].isnull()==False] #remove null JDs


from bs4 import BeautifulSoup

'''remove html tags and lower case'''
def plainSoup(entry):
    soup = BeautifulSoup(entry, 'lxml')
    soupText = soup.get_text().lower()
    soupText = soupText.replace("\n", " ")
    
    return soupText

mastercopy['JDSoup'] = mastercopy['JD'].apply(lambda a: plainSoup(a))

import nltk
from nltk.tokenize import RegexpTokenizer
from sumy.utils import get_stop_words
from nltk.stem import WordNetLemmatizer
stopwords = get_stop_words('english')
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()


'''tokenize and remove stop words and pos tag and lemmatize'''
def convertPos(x):
    if x.startswith('J'):
        return 'a'
    elif x.startswith('V'):
        return 'v'
    elif x.startswith('N'):
        return 'n'
    elif x.startswith('R'):
        return 'r'
    else:
        return 'n' #default

#to anothony: how can i remove postings with only arabic words? 
def filterSoup(x):
    tokens = tokenizer.tokenize(x)
    tokens = [i for i in tokens if i not in stopwords]
    tokens = nltk.pos_tag(tokens)
    
    cleantokens = [(lemmatizer.lemmatize(w,pos=convertPos(t))) for (w,t) in tokens]
    return cleantokens

mastercopy['JDtokens']= mastercopy['JDSoup'].apply(lambda a: filterSoup(a))

'''find key topics in each JD'''

from gensim import corpora, models, similarities

texts = [row for row in mastercopy.JDtokens]

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=2, no_above=0.8)
dictionary.save('jdtoke.dict')  # store the dictionary, for future reference
corpus = [dictionary.doc2bow(text) for text in texts] #change each JD to vector of word freq

#To anthony: how should id etermine chunksize or number of passes? 19K postings, 35 occupation categories
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=50, id2word = dictionary, chunksize= 500, passes=100)            
