import os
import math
import nltk
import pandas as pd
import numpy as np
import time
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import csv
def cosine_similarity(x, y):
    x_sqrt = np.sqrt(np.dot(x, x))
    y_sqrt = np.sqrt(np.dot(y, y))
    if y_sqrt != 0:     
        return (np.dot(x,y.T) / (x_sqrt * y_sqrt))
    elif y_sqrt == 0:
        return 0
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems
doc2vocab = dict()
vocab2doc = dict()
tnom=[]
tcontact=[]
with open("candidat4.csv", 'r', errors="ignore") as file:
        reader = csv.reader(file, delimiter=';')
        sent_score = []
        competence=[]
        nm=-1
        for eachline in reader:
            tnom.append(eachline[1])
            tcontact.append(eachline[4])
            nm+=1
            doc2vocab[nm] = dict()
            liste = eachline[3].split(',')
            score = 0
            for i in range(len(liste)):
                liste[i] = liste[i].replace('\'','')
                liste[i] = liste[i].replace('[','')
                liste[i] = liste[i].replace(']','')
                liste[i] = liste[i].replace(' ','')
                liste[i]= liste[i].lower()
                tokens = nltk.word_tokenize(liste[i])
                stop = set(stopwords.words('english'))
                tokens = [j for j in tokens if j not in stop] 
                for words in tokens:
                    if words in doc2vocab[nm]:
                        doc2vocab[nm][words] += 1
                    else:
                        doc2vocab[nm][words] = 1
                    
                    text_str = str(nm)
                    if words in vocab2doc:
                        if text_str not in vocab2doc[words]:
                            vocab2doc[words].append(text_str)
                            
                    else:
                        vocab2doc[words] = list()
                        vocab2doc[words].append(text_str)

term_pd = pd.DataFrame.from_dict(doc2vocab, orient='index')
term_pd = term_pd.fillna(0)

def doc_tf_idf(dataframe, query):
    
    # query tf-idf
    _, width = dataframe.shape
    final = list()
    
    # document tf-idf 
    new_tf = dataframe
    doc_term_value = dataframe[dataframe > 0].count().values # get array of number that document has that term
    doc_frequency = np.log(209 / (doc_term_value + 1))
    
    start = time.time()
    for i in range(len(dataframe)):
        results = np.zeros(width)
        one_row = dataframe.loc[i]
        row_value = one_row.values
        row_index = one_row.index
        
        for j,term in enumerate(row_index):
            if row_value[j] > 0:
                #term_frequency = 1 + np.log(row_value[j])
                term_frequency = np.log(row_value[j] + 1)
                new_tf.iloc[i,j] = term_frequency * doc_frequency[j]
                    
            elif row_value[j] == 0:
                term_frequency = 0
                new_tf.iloc[i,j] = 0
                
            if term in query:
                new_column = dataframe[[term]]
                new_col_value = new_column[new_column > 0].count().values
                results[j] = term_frequency * (np.log(209 / (new_col_value[0]+1)))
        final.append((i, cosine_similarity(new_tf.loc[i].values, results)))
    
        if i % 10 == 0:
            print ('step : %d, time : %f' % (i, time.time()-start))
            
    return new_tf, final
query = list()
f = open('./query.txt', 'r')
query = f.readlines()
query_token = nltk.word_tokenize(query[0])
term_doc_matrix, query_tf_idf = doc_tf_idf(term_pd, query_token)
print (term_doc_matrix[['python']])
score = sorted(query_tf_idf, key = lambda x : x[1], reverse=True)
print(score)