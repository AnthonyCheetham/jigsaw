import numpy as np
import pandas as pd
import glob
import psycopg2
import nltk
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import os
import matplotlib.pyplot as plt

def connect_to_db():
    try:
        conn = psycopg2.connect("dbname='mypuzzle' user=''"+os.environ['MYPUZZLE_UN']+\
                                "' host='localhost' password='"+os.environ['MYPUZZLE_PW']+"'")
        cursor = conn.cursor()
        print("Connected to database")
    except:
        print("Unable to connect to database!")
    return conn
            
##########
# Model file tools

def save_model(pkl_filename,model):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(pkl_filename):
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model

##########


def clean(review):
    # Remove unwanted characters
    for char in """(),"[]""":
        review = review.replace(char,'')
    review = review.replace('!','.')
    review = review.replace('?','.')
    review = review.replace('\n',' ')
    
    # And double spaces and ellipsis
    for ix in range(5):
        review = review.replace('  ',' ').replace('..','.')
    
    # lowercase
    review = review.lower()
    
    for x in ['puzzle','ravensburger','white mountain','puzzles']:
        review = review.replace(x,'')
    
    return review

##########
def plot_confusion_matrix(conf_matrix,cats=[],save=False,
                         cmap=plt.cm.Blues):
    n_classes = len(cats)
    
    plt.clf()
    plt.imshow(conf_matrix,cmap=cmap)
    fmt = '.2f'
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            if conf_matrix[i,j] > thresh:
                col = "white"
            else:
                col = 'black'
            
            plt.gca().text(j, i, format(conf_matrix[i, j], fmt),
                    ha="center", va="center",color=col)
    plt.xticks(np.arange(n_classes),cats,rotation=45,ha='right')
    plt.yticks(np.arange(n_classes),cats)
    plt.xlim(-0.5,n_classes-0.5)
    plt.ylim(n_classes-0.5,-0.5)
    plt.ylabel('True label')
    plt.xlabel('Predicted')
    plt.tight_layout()
    #
    if save:
        plt.savefig(save,dpi=300)
    plt.show()