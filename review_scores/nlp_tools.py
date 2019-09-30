import numpy as np
import pandas as pd
import glob
import psycopg2
import nltk
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

def connect_to_db():
    try:
        conn = psycopg2.connect("dbname='mypuzzle' user='postgres' host='localhost' password='gwv251'")
        cursor = conn.cursor()
        print("Connected to database")
    except:
        print("Unable to connect to database!")
    return conn

def get_reviews(conn,product_index=None):
    query = """SELECT review FROM reviews"""
    if product_index != None:
        query += """ WHERE product_index = """+str(product_index)
    cursor = conn.cursor()
    cursor.execute(query)
    all_reviews = cursor.fetchall()
    
    return all_reviews

def get_and_clean_reviews(conn,n_products = 6990):
    """ 
    """
    all_reviews = []
    for product_ix in np.arange(1,n_products+1):
        query = """SELECT review FROM reviews WHERE product_index = """+str(product_ix)
        cursor = conn.cursor()
        cursor.execute(query)
        product_reviews = cursor.fetchall()
        
        # Loop through them and clean them
        cleaned_product_reviews = []
        for review in product_reviews:
            if isinstance(review,tuple):
                temp = " ".join(x for x in review)
                review = temp
            clean_review = clean(review)
            
            # Break into sentences
            clean_sentences = clean_review.split('.')
            
            # Remove any sentences that are not really text
            for s in clean_sentences:
                if len(s) < 3:
                    clean_sentences.remove(s)
            
            cleaned_product_reviews.append(clean_sentences)
            
        all_reviews.append(cleaned_product_reviews)
    return all_reviews
            
##########
# Model file tools

def save_model(pkl_filename,model):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(pkl_filename):
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model

def train_lda_model(sentence_df,n_topics=5):
    """Run LDA using sklearn on the full corpus of sentences."""
    # Get the array for sklearn
    sentences = sentence_df['sentences'].values
    
    # Remove some brand names and repeated words
    cleaned_sentences = []
    for sentence in sentences:
        for x in ['puzzle','ravensburger','white mountain','puzzles']:
            sentence = sentence.replace(x,'')
        cleaned_sentences.append(sentence)
    print('Removed some uninformative words')

    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words='english')
    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(cleaned_sentences)

    # Helper function
    def print_topics(model, count_vectorizer, n_top_words):
        words = count_vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(model.components_):
            print("\nTopic #%d:" % topic_idx)
            print(" ".join([words[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))

    # Tweak the two parameters below
    number_words = 10
    # Create and fit the LDA model
    lda = LDA(n_components=n_topics, n_jobs=-1)
    lda.fit(count_data)
    # Print the topics found by the LDA model
    print("Topics found via LDA:")
    print_topics(lda, count_vectorizer, number_words)

    # Save the model
    save_model('lda_1000_'+str(n_topics)+'topics.pkl',lda)
    save_model('count_vectorizer.pkl',count_vectorizer)
    return lda,count_vectorizer

##########


def clean(review):
    # Remove stupid characters
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

def tokenize(review):
    # Tokenize
    review = nltk.word_tokenize(review)
    return review

def remove_stops(review):
    # Remove stop words
    en_stop = set(nltk.corpus.stopwords.words('english')) # set stop words
    
    # updates set of stop words with the ones I wanted to add (i.e., 
    # very frequently used emoticons)
    en_stop.update(['star','puzzle','ravensburger']) 
    
    # Remove stop words:
    review = [word for word in review if word not in en_stop]
    
    return review

def clean_tokenize_stops(review):
    """ Wrapper function to do all 3 preprocessing steps"""
    review = clean(review)
    review = tokenize(review)
    review = remove_stops(review)
    return review

#########
### Functions borrowed from Bart:
# Apply pre-trained word2vec model to a single word. 
def _evaluate(word,use_model=None):
        
    if(isinstance(word,list)):
        return __evaluate_set(word,use_model=use_model)
    elif(isinstance(word,str)):
        #attempt to get vectorial representation of word.
        try:
            return use_model[word]
        except KeyError as e:
            return np.full([300,],np.nan)
    else:
        raise TypeError()
            
# Apply the word2vec model to a set of words and average them. 
def __evaluate_set(words,use_model=None):
    #evaluate each word in 
    n = 0
    a = []
    for w in words:
        #attempt to evaluate vectorial representation of word.
        try:
            v = use_model[w]
            if((np.isnan(v).any() + np.isinf(v).any()) == 0):
                a.append(v)
                n += 1
        except KeyError as e:
            pass
    #if nothing was valid, return nan
    if(n==0):
        return np.full([300,], np.nan)
    #return average
    return np.mean(np.array(a),axis=0)