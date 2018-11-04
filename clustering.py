import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.matutils import corpus2csc
from gensim.utils import simple_preprocess
import re
import spacy
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from MulticoreTSNE import MulticoreTSNE as TSNE

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

def lda_sklearn(text,num_topics=10):
    #count_vect = CountVectorizer(max_features=100000)
    count_vect = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=50000,
                                    stop_words='english')
    X_counts = count_vect.fit_transform(text)
    #tfidf_tr = TfidfTransformer()
    #X_tfidf = tfidf_tr.fit_transform(X_counts)    
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,n_jobs=-1)
    lda.fit(X_counts)
    X_lda = lda.transform(X_counts)    
    tf_feature_names = count_vect.get_feature_names()    
    print_top_words(lda, tf_feature_names, 20)    
    return X_lda

def lda_gensim(texts,num_topics=10):
    id2word = Dictionary(texts)
    corpus = [id2word.doc2bow(text) for text in texts]  
    lda = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
    for top in lda.print_topics():
        print(top)
    lda_corpus = lda[corpus]
    X_lda = corpus2csc(lda_corpus).todense().T
    return X_lda

def process_text(row):
    # convert text to lowercase
    text = row.lower()
    # sub common symbols and remove non-ascii chars
    text = re.sub(r'[.]{2,}',' ',text)
    text = text.encode("ascii", errors="ignore").decode()
    # remove all punctuation except apostrophes and dollar sign
    text = re.sub(r"[^\w\s'\$]",'',text)
    text = re.sub(r'[ ]{2,}',' ',text)
    #text = text.translate(str.maketrans('', '', string.punctuation))
    # remove stop words
    stops = stopwords.words('english')
    text = ' '.join(word for word in text.split() if word not in stops)
    # stem words
#    stemmer = SnowballStemmer("english")
#    stemmed_text = ''
#    for word in text.split():
#            stemmed_text += (stemmer.stem(word))+' '
#    text = stemmed_text
    return text

def tokenize(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    tokens = ' '.join(token.text for token in doc)
    return tokens

def plot_clusters(X,y=None,model_type=None):
    _,ax = plt.subplots(figsize=(8,8))
    if model_type is None or y is None:
        ax.scatter(X[:,0],X[:,1], s=1.)
    else:
        num_clusters = len(np.unique(y))
        for i in range(num_clusters):
            if not np.any(y == i):
                continue
            ax.scatter(X[y == i, 0], X[y == i, 1], s=1., cmap='Dark2')
        ax.set_title(model_type+': '+str(num_clusters)+' clusters')
    plt.show()

df = load_data()
df = df[df.is_justext==1]
#df['text_processed'] = df.text.apply(process_text)

stop_words = stopwords.words('english')
df['text_processed'] = df.text.apply(lambda x: ' '.join(word for word in simple_preprocess(x)
                                                        if word not in stop_words))

nlp = spacy.load('en_core_web_sm')
df['tokens'] = df.text_processed.apply(lambda x:' '.join(t.text for t in nlp.tokenizer(x)))

num_topics = 10

texts = [row.split() for row in df.tokens]
X_lda = lda_gensim(texts,num_topics)

#X_lda = lda_sklearn(df.tokens,num_topics)

#model = KMeans(n_clusters=num_topics,n_jobs=-1)
#model.fit(X_lda)
#yhat = model.predict(X_lda)
#
#df['clusters'] = yhat
#
#tsne = TSNE(n_jobs=4)
#X_tsne = tsne.fit_transform(X_lda)
#plot_clusters(X_tsne,yhat,'k-means')

