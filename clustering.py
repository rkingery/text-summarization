import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from gensim import corpora, models, similarities
import re
import string
import spacy
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

np.random.seed(23)

def load_data():
    data_path = Path('./data')
    print('loading ./data/html_and_text_big.csv into dataframe')
    df = pd.read_csv(data_path/'html_and_text_big.csv',index_col=0)
    print('done')
    return df
    
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

def lda_sklearn(text):
    #count_vect = CountVectorizer(max_features=100000)
    count_vect = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=50000,
                                    stop_words='english')
    X_counts = count_vect.fit_transform(text)
    #tfidf_tr = TfidfTransformer()
    #X_tfidf = tfidf_tr.fit_transform(X_counts)
    
    lda = LatentDirichletAllocation(n_components=10, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,n_jobs=-1)
    lda.fit(X_counts)
    
    tf_feature_names = count_vect.get_feature_names()
    
    print_top_words(lda, tf_feature_names, 20)

def process_text(row):
    # convert text to lowercase
    text = row.text.lower()
    # sub common symbols and remove non-ascii chars
    #text = text.replace('&',' and ')
    #text = text.replace('£','$ ')
    text = re.sub(r'[ ]{2,}',' ',text)
    text = re.sub(r'[.]{2,}',' ',text)
    text = text.encode("ascii", errors="ignore").decode()
    # remove all punctuation except apostrophes and dollar sign
    text = re.sub(r"[^\w\s'\$]",'',text)
    #text = text.translate(str.maketrans('', '', string.punctuation))
    # remove stop words
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    # stem words
#    stemmer = SnowballStemmer("english")
#    stemmed_text = ''
#    for word in text.split():
#            stemmed_text += (stemmer.stem(word))+' '
#    text = stemmed_text
    return text

def tokenize(text,nlp):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    tokens = ' '.join(token.text for token in doc)
    return tokens

df = load_data()
df = df[df.is_justext==1]
#df['text_processed'] = df.text.apply(process_text)
#nlp = spacy.load('en_core_web_sm')
#df['tokens'] = df.text_processed.apply(lambda x:' '.join(t.text for t in nlp.tokenizer(x)))