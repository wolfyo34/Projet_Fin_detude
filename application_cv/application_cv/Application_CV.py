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
from tkinter import *
from tkinter import ttk
import tkinter.font as font

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras import metrics
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.layers import Flatten

from nltk import word_tokenize, pos_tag, chunk
from pprint import pprint
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

from pprint import pprint
import pandas as pd
import numpy as np

from keras import optimizers
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.constraints import maxnorm
from keras.layers import Dropout
import os
import csv


fenetre = Tk()
fenetre.title("job/candidate search application")
fenetre.geometry("640x480")

f = font.Font(size=50)
f1 = font.Font(size=20)

doc2vocab = dict()
vocab2doc = dict()
tnom=[]
tcontact=[]
description2=""
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
        cand=reader.line_num
term_pd = pd.DataFrame.from_dict(doc2vocab, orient='index')
term_pd = term_pd.fillna(0)
doc2vocab = dict()
vocab2doc = dict()
tnom2=[]
tcontact2=[]
tposte=[]
u=0
with open("jobs6.csv", 'r', errors="ignore") as file:
        reader = csv.reader(file, delimiter=';')
        sent_score2 = []
        competence2=[]
        nm=-1
        job2n=0
        for eachline in reader:
            liste2 = eachline[6].split(',')
            liste3 = eachline[4].split(' ')
            score2 = 0
            liste3= liste3[0].split('/')
            if u==0:
                u+=1
            else:
                if liste3[0] != "Apply":   
                    salaire = float(liste3[0][1:].replace(',',''))
                    if salaire < 1000:
                        salaire*=2000
                    if salaire >= 50000:
                        print(salaire)
                        tnom2.append(eachline[1])
                        tcontact2.append(eachline[7])
                        tposte.append(eachline[2])
        
                        job2n+=1
                        nm+=1
                        doc2vocab[nm] = dict()
                        for i in range(len(liste2)):
                            liste2[i] = liste2[i].replace('\'','')
                            liste2[i] = liste2[i].replace('[','')
                            liste2[i] = liste2[i].replace(']','')
                            liste2[i] = liste2[i].replace(' ','')
                            liste2[i]= liste2[i].lower()
                            tokens = nltk.word_tokenize(liste2[i])
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
term_pd2 = pd.DataFrame.from_dict(doc2vocab, orient='index')
term_pd2 = term_pd2.fillna(0)

###########################################################################################
tabguillement=[]
with open("jobs8.csv", 'r', errors="ignore") as file:
        reader = csv.reader(file, delimiter=';')
        for eachline in reader:
            for i in range(len(eachline)):
                eachline[i]="\""+eachline[i]+"\""
            tabguillement.append(eachline)
columns = ["ID","Nom","Query","place","salaire","Description","competence","url"]
data = pd.DataFrame(data=tabguillement,columns=columns)

train, test = train_test_split(data, test_size = 0.1)

train_descs = train['Description']
train_labels = train['Query']
#train_labels = train['Job Title']
 
test_descs = test['Description']
test_labels = test['Query']
num_labels = len(train_labels.unique().tolist())
vocab_size = 1000
batch_size = 32
nb_epoch = 30

# define Tokenizer with Vocab Size
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_descs)
x_train = tokenizer.texts_to_matrix(train_descs, mode='tfidf')
x_test = tokenizer.texts_to_matrix(test_descs, mode='tfidf')
 
encoder = LabelBinarizer()
encoder.fit(train_labels)
y_train = encoder.transform(train_labels)
y_test = encoder.transform(test_labels)
model = Sequential()
model.add(Dense(4096, input_shape=(vocab_size,), activation = 'relu', kernel_initializer = 'glorot_normal', kernel_constraint=maxnorm(2)))
model.add(Dropout(0.1))
model.add(Dense(1024, kernel_initializer = 'glorot_normal', activation= 'relu'))
model.add(Dropout(0.1))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
    
    # Compile model
model.compile(loss = 'categorical_crossentropy',optimizer = 'sgd',metrics = [metrics.categorical_accuracy, 'accuracy'])
history = model.fit(x_train, y_train,batch_size=batch_size,epochs=nb_epoch,verbose=1,validation_split=0.1)
# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
 
print('\nTest categorical_crossentropy:', score[0])
print('Categorical accuracy:', score[1])
print('Accuracy:', score[2])

def Prediction(model,user_text):
    
    # Encode the text
    encoded_docs = [one_hot(user_text, vocab_size)]
    
    # pad documents to a max length
    padded_text = pad_sequences(encoded_docs, maxlen=1000, padding='post')
    
    #Prediction based on model
    prediction = model.predict(padded_text)
    
    #Decode the prediction
    encoder = LabelBinarizer()
    encoder.fit(test_labels)
    result = encoder.inverse_transform(prediction)
    
    return result[0]
    
############################################################################################################################

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


def doc_tf_idf(dataframe, query,taille):
    
    # query tf-idf
    _, width = dataframe.shape
    final = list()
    
    # document tf-idf 
    new_tf = dataframe
    doc_term_value = dataframe[dataframe > 0].count().values # get array of number that document has that term
    doc_frequency = np.log((taille-1) / (doc_term_value + 1))
    
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
                results[j] = term_frequency * (np.log((taille-1) / (new_col_value[0]+1)))
        final.append((i, cosine_similarity(new_tf.loc[i].values, results)))
    
        if i % 10 == 0:
            print ('step : %d, time : %f' % (i, time.time()-start))
            
    return new_tf, final
def window_entreprise3():
    global query
    query=[]
    fenetre2 = Tk()
    fenetre2.title("job/candidate search application")
    fenetre2.geometry("640x480")
    
    if int(entrée99.get()) >=10:
        if len(entrée100.get()) !=0:
            query.append(entrée100.get())
    if int(entrée99.get()) >=9:
        if len(entrée90.get()) !=0:
            query.append(entrée90.get())
    if int(entrée99.get()) >=8:
        if len(entrée80.get()) !=0:
            query.append(entrée80.get())
    if int(entrée99.get()) >=7:
        if len(entrée70.get()) !=0:
            query.append(entrée70.get())
    if int(entrée99.get()) >=6:
        if len(entrée60.get()) !=0:
            query.append(entrée60.get())
    if int(entrée99.get()) >=5:
        if len(entrée50.get()) !=0:
            query.append(entrée50.get())
    if int(entrée99.get()) >=4:
        if len(entrée40.get()) !=0:
            query.append(entrée40.get())
    if int(entrée99.get()) >=3:
        if len(entrée30.get()) !=0:
            query.append(entrée30.get())
    if int(entrée99.get()) >=2:
        if len(entrée20.get()) !=0:
            query.append(entrée20.get())
    if int(entrée99.get()) >=1:
        if len(entrée10.get()) !=0:
            query.append(entrée10.get())
            
    term_doc_matrix, query_tf_idf = doc_tf_idf(term_pd, query,cand)
    score = sorted(query_tf_idf, key = lambda x : x[1], reverse=True)
    print(score)
    labelExample = Label(fenetre2, text = "RECAP :")
    
    labelExample['font'] = f
    
    labelExample.pack(side=TOP, padx=0, pady=0)
    
    tv = ttk.Treeview(fenetre2)
    tv['columns']=('Rank', 'Name','contact', 'pourcentage')
    tv.column('#0', width=0, stretch=NO)
    tv.column('Rank', anchor=CENTER, width=100)
    tv.column('Name', anchor=CENTER, width=100)
    tv.column('contact', anchor=CENTER, width=350)
    tv.column('pourcentage', anchor=CENTER, width=50)
    
    tv.heading('#0', text='', anchor=CENTER)
    tv.heading('Rank', text='Id', anchor=CENTER)
    tv.heading('Name', text='Candidate', anchor=CENTER)
    tv.heading('contact', text='Contact', anchor=CENTER)
    tv.heading('pourcentage', text='%', anchor=CENTER)
    tv.insert(parent='', index=0, iid=0, text='', values=('1',tnom[score[0][0]],tcontact[score[0][0]],round(score[0][1]*100),2))
    tv.insert(parent='', index=1, iid=1, text='', values=('2',tnom[score[1][0]],tcontact[score[1][0]],round(score[1][1]*100),2))
    tv.insert(parent='', index=2, iid=2, text='', values=('3',tnom[score[2][0]],tcontact[score[2][0]],round(score[2][1]*100),2))
    tv.insert(parent='', index=3, iid=3, text='', values=('4',tnom[score[3][0]],tcontact[score[3][0]],round(score[3][1]*100),2))
    tv.insert(parent='', index=4, iid=4, text='', values=('5',tnom[score[4][0]],tcontact[score[4][0]],round(score[4][1]*100),2))
    tv.pack(side=TOP, padx=0, pady=40)


def window_entreprise2():
    global entrée10,entrée20,entrée30,entrée40,entrée50,entrée60,entrée70,entrée80,entrée90,entrée100

    fenetre1 = Tk()
    fenetre1.title("job/candidate search application")
    fenetre1.geometry("640x480")
    
    labelExample = Label(fenetre1, text = "SKILLS")
    labelExample.pack(side=TOP, padx=0, pady=0)

    if int(entrée99.get())>=1:
        entrée10 = Entry (fenetre1, width=40)
        entrée10.pack(side=TOP, padx=0, pady=12)
    if int(entrée99.get())>=2:
        entrée20 = Entry (fenetre1, width=40)
        entrée20.pack(side=TOP, padx=0, pady=12)
    if int(entrée99.get())>=3:
        entrée30 = Entry (fenetre1, width=40)
        entrée30.pack(side=TOP, padx=0, pady=12)
    if int(entrée99.get())>=4:
        entrée40 = Entry (fenetre1, width=40)
        entrée40.pack(side=TOP, padx=0, pady=12)
    if int(entrée99.get())>=5:
        entrée50 = Entry (fenetre1, width=40)
        entrée50.pack(side=TOP, padx=0, pady=12)     
    if int(entrée99.get())>=6:
        entrée60 = Entry (fenetre1, width=40)
        entrée60.pack(side=TOP, padx=0, pady=12)
    if int(entrée99.get())>=7:
        entrée70 = Entry (fenetre1, width=40)
        entrée70.pack(side=TOP, padx=0, pady=12)
    if int(entrée99.get())>=8:
        entrée80 = Entry (fenetre1, width=40)
        entrée80.pack(side=TOP, padx=0, pady=12)
    if int(entrée99.get())>=9:
        entrée90 = Entry (fenetre1, width=40)
        entrée90.pack(side=TOP, padx=0, pady=12)
    if int(entrée99.get())>=10:
        entrée100 = Entry (fenetre1, width=40)
        entrée100.pack(side=TOP, padx=0, pady=12)





    boutonval_1 = Button (fenetre1, text = "Validate",bg = "green",fg = "white",command=window_entreprise3)
    labelExample['font'] = f
    boutonval_1['font'] = f1    

    boutonval_1.pack(side=TOP, padx=0, pady=20)
    
def window_entreprise1():
    global entrée99,entrée98

    fenetre1 = Tk()
    fenetre1.title("job/candidate search application")
    fenetre1.geometry("640x480")
    
    labelExample = Label(fenetre1, text = "NUMBER OF SKILLS SEARCH")
    entrée99 = Entry (fenetre1, width=40)
    labelExample1 = Label(fenetre1, text = "SALARY")
    entrée98 = Entry (fenetre1, width=40)
    

    boutonval_1 = Button (fenetre1, text = "Validate",bg = "green",fg = "white",command=window_candidat2)
    labelExample['font'] = f
    boutonval_1['font'] = f1    

    labelExample.pack(side=TOP, padx=0, pady=0)
    entrée99.pack(side=TOP, padx=0, pady=12)
    labelExample1.pack(side=TOP, padx=0, pady=12)
    entrée98.pack(side=TOP, padx=0, pady=12)
    boutonval_1.pack(side=TOP, padx=0, pady=20)

   
    
def window_candidat3():
    global query
    query=[]
    fenetre2 = Tk()
    fenetre2.title("job/candidate search application")
    fenetre2.geometry("640x480") 
    
    if int(entrée99.get()) >=10:
        if len(entrée100.get()) !=0:
            query.append(entrée100.get())
    if int(entrée99.get()) >=9:
        if len(entrée90.get()) !=0:
            query.append(entrée90.get())
    if int(entrée99.get()) >=8:
        if len(entrée80.get()) !=0:
            query.append(entrée80.get())
    if int(entrée99.get()) >=7:
        if len(entrée70.get()) !=0:
            query.append(entrée70.get())
    if int(entrée99.get()) >=6:
        if len(entrée60.get()) !=0:
            query.append(entrée60.get())
    if int(entrée99.get()) >=5:
        if len(entrée50.get()) !=0:
            query.append(entrée50.get())
    if int(entrée99.get()) >=4:
        if len(entrée40.get()) !=0:
            query.append(entrée40.get())
    if int(entrée99.get()) >=3:
        if len(entrée30.get()) !=0:
            query.append(entrée30.get())
    if int(entrée99.get()) >=2:
        if len(entrée20.get()) !=0:
            query.append(entrée20.get())
    if int(entrée99.get()) >=1:
        if len(entrée10.get()) !=0:
            query.append(entrée10.get())

    print(1)
    term_doc_matrix, query_tf_idf = doc_tf_idf(term_pd2, query,job2n)
    print(2)
    score = sorted(query_tf_idf, key = lambda x : x[1], reverse=True)  
    print(score)
    labelExample = Label(fenetre2, text = "RECAP :")
    
    labelExample['font'] = f
    
    labelExample.pack(side=TOP, padx=0, pady=0)
    
    tv = ttk.Treeview(fenetre2)
    tv['columns']=('Rank', 'Name','Poste','contact', 'pourcentage')
    tv.column('#0', width=0, stretch=NO)
    tv.column('Rank', anchor=CENTER, width=50)
    tv.column('Name', anchor=CENTER, width=100)
    tv.column('Poste', anchor=CENTER, width=100)
    tv.column('contact', anchor=CENTER, width=300)
    tv.column('pourcentage', anchor=CENTER, width=50)
    
    tv.heading('#0', text='', anchor=CENTER)
    tv.heading('Rank', text='Id', anchor=CENTER)
    tv.heading('Name', text='business', anchor=CENTER)
    tv.heading('Poste', text='job', anchor=CENTER)
    tv.heading('contact', text='Contact', anchor=CENTER)
    tv.heading('pourcentage', text='%', anchor=CENTER)
    print("ici",term_pd2.index[7])
    tv.insert(parent='', index=0, iid=0, text='', values=('1',tnom2[term_pd2.index[score[0][0]]],tposte[term_pd2.index[score[0][0]]],tcontact2[term_pd2.index[score[0][0]]],round(score[0][1]*100),2))
    tv.insert(parent='', index=1, iid=1, text='', values=('2',tnom2[term_pd2.index[score[1][0]]],tposte[term_pd2.index[score[1][0]]],tcontact2[term_pd2.index[score[1][0]]],round(score[1][1]*100),2))
    tv.insert(parent='', index=2, iid=2, text='', values=('3',tnom2[term_pd2.index[score[2][0]]],tposte[term_pd2.index[score[2][0]]],tcontact2[term_pd2.index[score[2][0]]],round(score[2][1]*100),2))
    tv.insert(parent='', index=3, iid=3, text='', values=('4',tnom2[term_pd2.index[score[3][0]]],tposte[term_pd2.index[score[3][0]]],tcontact2[term_pd2.index[score[3][0]]],round(score[3][1]*100),2))
    tv.insert(parent='', index=4, iid=4, text='', values=('5',tnom2[term_pd2.index[score[4][0]]],tposte[term_pd2.index[score[4][0]]],tcontact2[term_pd2.index[score[4][0]]],round(score[4][1]*100),2))
    tv.pack(side=TOP, padx=0, pady=40)
    
    
def window_candidat2():
    global entrée10,entrée20,entrée30,entrée40,entrée50,entrée60,entrée70,entrée80,entrée90,entrée100

    fenetre1 = Tk()
    fenetre1.title("job/candidate search application")
    fenetre1.geometry("640x480")
    
    labelExample = Label(fenetre1, text = "SKILLS")
    labelExample.pack(side=TOP, padx=0, pady=0)

    if int(entrée99.get())>=1:
        entrée10 = Entry (fenetre1, width=40)
        entrée10.pack(side=TOP, padx=0, pady=12)
    if int(entrée99.get())>=2:
        entrée20 = Entry (fenetre1, width=40)
        entrée20.pack(side=TOP, padx=0, pady=12)
    if int(entrée99.get())>=3:
        entrée30 = Entry (fenetre1, width=40)
        entrée30.pack(side=TOP, padx=0, pady=12)
    if int(entrée99.get())>=4:
        entrée40 = Entry (fenetre1, width=40)
        entrée40.pack(side=TOP, padx=0, pady=12)
    if int(entrée99.get())>=5:
        entrée50 = Entry (fenetre1, width=40)
        entrée50.pack(side=TOP, padx=0, pady=12)     
    if int(entrée99.get())>=6:
        entrée60 = Entry (fenetre1, width=40)
        entrée60.pack(side=TOP, padx=0, pady=12)
    if int(entrée99.get())>=7:
        entrée70 = Entry (fenetre1, width=40)
        entrée70.pack(side=TOP, padx=0, pady=12)
    if int(entrée99.get())>=8:
        entrée80 = Entry (fenetre1, width=40)
        entrée80.pack(side=TOP, padx=0, pady=12)
    if int(entrée99.get())>=9:
        entrée90 = Entry (fenetre1, width=40)
        entrée90.pack(side=TOP, padx=0, pady=12)
    if int(entrée99.get())>=10:
        entrée100 = Entry (fenetre1, width=40)
        entrée100.pack(side=TOP, padx=0, pady=12)



    boutonval_1 = Button (fenetre1, text = "Validate",bg = "green",fg = "white",command=window_candidat3)
    
    labelExample['font'] = f
    boutonval_1['font'] = f1
    
    labelExample.pack(side=TOP, padx=0, pady=0)
    boutonval_1.pack(side=TOP, padx=0, pady=20)
    
def window_candidat1():
    global entrée99,entrée98

    fenetre1 = Tk()
    fenetre1.title("job/candidate search application")
    fenetre1.geometry("640x480")
    
    labelExample = Label(fenetre1, text = "NUMBER OF SKILLS SEARCH")
    entrée99 = Entry (fenetre1, width=40)
    labelExample1 = Label(fenetre1, text = "SALARY")
    entrée98 = Entry (fenetre1, width=40)
    

    boutonval_1 = Button (fenetre1, text = "Validate",bg = "green",fg = "white",command=window_candidat2)
    labelExample['font'] = f
    boutonval_1['font'] = f1    

    labelExample.pack(side=TOP, padx=0, pady=0)
    entrée99.pack(side=TOP, padx=0, pady=12)
    labelExample1.pack(side=TOP, padx=0, pady=12)
    entrée98.pack(side=TOP, padx=0, pady=12)
    boutonval_1.pack(side=TOP, padx=0, pady=20)
    
def window_menu():
    fenetre1 = Tk()
    fenetre1.title("job/candidate search application")
    fenetre1.geometry("640x480")

    bouton1 = Button (fenetre1,text = "company",bg = "gray",fg = "white", width=55,command=window_entreprise1)
    bouton2 = Button (fenetre1,text = "candidate",bg = "gray",fg = "white", width=55,command=window_candidat1)
    bouton1['font'] = f
    bouton2['font'] = f
    bouton1.pack(side=TOP, padx=0, pady=55)
    bouton2.pack(side=BOTTOM, padx=0, pady=55)
    
#####             PARTIE FONCTIONELLE           ########
def window_entreprise3bis():
    fenetre2 = Tk()
    fenetre2.title("job/candidate search application")
    fenetre2.geometry("640x480")

    print(entrée10.get("1.0",'end-1c'))

    kirikou = Prediction(model,entrée10.get("1.0",'end-1c'))
    kirikou=kirikou.replace('\"','')
    print(kirikou)
 
    with open("jobs8.csv", 'r', errors="ignore") as file:
            reader = csv.reader(file, delimiter=';')
            for eachline in reader:
                for i in range(len(eachline)):
                    if(eachline[i]==kirikou):
                        nom2=eachline[1]
                        salaire2=eachline[4]
                        url2=eachline[7]

                    
    
    labelExample = Label(fenetre2, text = "RECAP")
    
    labelExample['font'] = f
    
    labelExample.pack(side=TOP, padx=0, pady=0)
    
    tv = ttk.Treeview(fenetre2)
    tv['columns']=('Rank', 'Name','nom','salaire','url')
    tv.column('#0', width=0, stretch=NO)
    tv.column('Rank', anchor=CENTER, width=100)
    tv.column('Name', anchor=CENTER, width=150)
    tv.column('nom', anchor=CENTER, width=100)
    tv.column('salaire', anchor=CENTER, width=100)
    tv.column('url', anchor=CENTER, width=100)
    
    tv.heading('#0', text='', anchor=CENTER)
    tv.heading('Rank', text='Id', anchor=CENTER)
    tv.heading('Name', text='job offer', anchor=CENTER)
    tv.heading('nom', text='name', anchor=CENTER)
    tv.heading('salaire', text='salary', anchor=CENTER)
    tv.heading('url', text='url', anchor=CENTER)
    

    
    tv.insert(parent='', index=0, iid=0, text='', values=('1',kirikou,nom2,salaire2,url2))
    tv.pack(side=TOP, padx=0, pady=40)


def window_entreprise2bis():
    global entrée10

    fenetre1 = Tk()
    fenetre1.title("job/candidate search application")
    fenetre1.geometry("640x480")
    
    labelExample = Label(fenetre1, text = "DESCRIPTION")
    labelExample.pack(side=TOP, padx=0, pady=0)


    entrée10 = Text (fenetre1, width=65)
    entrée10.pack(side=TOP, padx=0, pady=0)

    boutonval_1 = Button (fenetre1, text = "Validate",bg = "green",fg = "white",command=window_entreprise3bis)
    labelExample['font'] = f
    boutonval_1['font'] = f1    

    boutonval_1.pack(side=TOP, padx=0, pady=20)   
    
def window_candidat3bis():
    fenetre2 = Tk()
    fenetre2.title("job/candidate search application")
    fenetre2.geometry("640x480")

    kirikou = Prediction(model,entrée10.get("1.0",'end-1c'))
    kirikou=kirikou.replace('\"','')
    print(kirikou)
 
    with open("candidat40.csv", 'r', errors="ignore") as file:
            reader = csv.reader(file, delimiter=';')
            for eachline in reader:
                for i in range(len(eachline)):
                    if(eachline[i]==kirikou):
                        nom20=eachline[1]
                        print(nom20 + 'nom')
                        description20=eachline[4]
                        url20=eachline[4]
                        print(url20 + 'url')

                    
    
    labelExample = Label(fenetre2, text = "RECAP")
    
    labelExample['font'] = f
    
    labelExample.pack(side=TOP, padx=0, pady=0)
    
    tv = ttk.Treeview(fenetre2)
    tv['columns']=('Rank', 'Name','nom','description','url')
    tv.column('#0', width=0, stretch=NO)
    tv.column('Rank', anchor=CENTER, width=100)
    tv.column('Name', anchor=CENTER, width=150)
    tv.column('nom', anchor=CENTER, width=100)
    tv.column('description', anchor=CENTER, width=100)
    tv.column('url', anchor=CENTER, width=100)
    
    tv.heading('#0', text='', anchor=CENTER)
    tv.heading('Rank', text='Id', anchor=CENTER)
    tv.heading('Name', text='job offer', anchor=CENTER)
    tv.heading('nom', text='name', anchor=CENTER)
    tv.heading('description', text='description', anchor=CENTER)
    tv.heading('url', text='url', anchor=CENTER)
    

    
    tv.insert(parent='', index=0, iid=0, text='', values=('1',kirikou,nom20,description20,url20))
    tv.pack(side=TOP, padx=0, pady=40)
    
def window_candidat2bis():
    global entrée10

    fenetre1 = Tk()
    fenetre1.title("job/candidate search application")
    fenetre1.geometry("640x480")
    
    labelExample = Label(fenetre1, text = "DESCRIPTION")
    labelExample.pack(side=TOP, padx=0, pady=0)


    entrée10 = Text (fenetre1, width=65)
    entrée10.pack(side=TOP, padx=0, pady=0)

    boutonval_1 = Button (fenetre1, text = "Validate",bg = "green",fg = "white",command=window_candidat3bis)
    labelExample['font'] = f
    boutonval_1['font'] = f1    

    boutonval_1.pack(side=TOP, padx=0, pady=20)   
    
    
    
def window_menubis():
    fenetre1 = Tk()
    fenetre1.title("job/candidate search application")
    fenetre1.geometry("640x480")

    bouton1 = Button (fenetre1,text = "company ",bg = "gray",fg = "white", width=55,command=window_entreprise2bis)
    bouton2 = Button (fenetre1,text = "candidate",bg = "gray",fg = "white", width=55,command=window_candidat2bis)
    
    bouton1['font'] = f
    bouton2['font'] = f
    bouton1.pack(side=TOP, padx=0, pady=55)
    bouton2.pack(side=BOTTOM, padx=0, pady=55)




bouton1 = Button (fenetre,text = "content based",bg = "gray",fg = "white",command=window_menu)
bouton2 = Button (fenetre,text = "collaborative filtering",bg = "gray",fg = "white",command=window_menubis)
bouton1['font'] = f
bouton2['font'] = f
bouton1.pack(side=TOP, padx=0, pady=55)
bouton2.pack(side=BOTTOM, padx=0, pady=55)

fenetre.mainloop()

