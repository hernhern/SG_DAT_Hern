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
        
def filterSoup(x):
    tokens = tokenizer.tokenize(x)
    tokens = [i for i in tokens if i not in stopwords]
    tokens = nltk.pos_tag(tokens)
    
    cleantokens = [(lemmatizer.lemmatize(w,pos=convertPos(t))) for (w,t) in tokens]
    return cleantokens

mastercopy['JDtokens']= mastercopy['JDSoup'].apply(lambda a: filterSoup(a))

from gensim import corpora, models

dictionary = corpora.Dictionary(texts)
  
def findtopic(jobdesc):
    '''LDA'''
    

#df.loc[df['A'] == 'foo']            
'''check rship between pay and popularity'''
#tempMatrix = mastercopy[['jobCat1', 'minSalary', 'maxSalary', 'postViews', 'totalApp']] 
#for x in mastercopy.jobCat1.unique():
#    occMat = tempMatrix.loc[tempMatrix['jobCat1']==x] #get matrix with occ
#    occMattemp = occMat[['minSalary', 'maxSalary', 'postViews','totalApp']] #num values only
#    print(x)
#    print(occMattemp.corr())
#conclusion: some occupation has stronger correlation
    
                         
#mastercopy = mastercopy[mastercopy['JD'].str.replace('[^A-Za-z\s]+', '')]
#remove arabic JDs

"separate date and time of app"

#appDate = [] #initiate appDate list
#appTime = [] #initiate appTime list
#
#for i in apps.app_date:
#    appDate.append(i.split(' ')[0])
#    appTime.append(i.split(' ')[1])
#    
#apps['appDate']=appDate
#apps['appTime']=appTime

##from sumy.parsers.html import HtmlParser
##from sumy.nlp.tokenizers import Tokenizer
#from sumy.summarizers.lsa import LsaSummarizer
##from sumy.summarizers import LexRankSummarizer
#from sumy.nlp.stemmers.czech import stem_word
#from sumy.utils import get_stop_words



#testCols = ['minSalary', 'maxSalary', 'vacancies', 'totalApp']
#testX = mastersheet[testCols].ix[0:100,:]
##testX.drop_na

#from sklearn.linear_model import LinearRegression
#
###linreg = LinearRegression()
##featurecols = ['minSalary', 'maxSalary', 'vacancies','totalApp']
###y = mastersheet.totalApp
##
##X = mastersheet[featurecols]
##
###y_pred = linreg.fit(X,y)



  
#'''hypotheses
#1. How do jobseekers apply? Regularly over days or all at once?
#2. Are there jobseekers who apply to the same few jobs? 
#3. Can we recommend them similar jobs? content-based filtering.'''#Rfrom nltk.tokenize import egexpTokenizer