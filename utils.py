import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
from justext import justext, get_stoplist

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phraser
from gensim.models import Phrases
import spacy

from sklearn.cluster import KMeans
from sklearn.externals import joblib

global spam
spam = ['advertisement', 'getty', 'published', 'javascript', 'updated', 'jpghttps',
        'posted', 'read more', 'photo gallery', 'play video', 'caption']

def clean_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    if soup.body is None:
        return None
    for tag in soup.body.select('script'):
        tag.decompose()
    for tag in soup.body.select('style'):
        tag.decompose()
    return str(soup)

def get_text_bs(html):
    soup = BeautifulSoup(html, 'html.parser')
    body = soup.body
    if body is None:
        return None
    for tag in body.select('script'):
        tag.decompose()
    for tag in body.select('style'):
        tag.decompose()
    text = body.get_text(separator='\n')
    return text

def filter_pages(text):
    if not isinstance(text,str):
        print(text)
    for f in spam:
        if f in text.lower() and len(text)<100:
            #print('spam found:',f)
            return False
    return True

def get_text_jt(html):
    text = []
    paragraphs = justext(html, get_stoplist("English"))
    for paragraph in paragraphs:
        if not paragraph.is_boilerplate:
            if len(paragraph.text) > 15 and filter_pages(paragraph.text):
                text.append(paragraph.text)
            #else:
            #    print(len(paragraph.text),' :: ',paragraph.text)
    text = ' '.join(t for t in text)
    return text
    
def process_warc(file_path):
    rows = []
    dropped_count = 0
    with open(file_path, 'rb') as stream:
        for record in tqdm(ArchiveIterator(stream)):
            try:
                if record.rec_type == 'response':
                    url = record.rec_headers.get_header('WARC-Target-URI')
                    html_raw = record.content_stream().read()
                    html = html_raw.decode('utf-8')
                    html = clean_html(html)
                    text = get_text_jt(html_raw)
                    rows.append([url,html,text])
            except:
                dropped_count += 1
                #print(dropped_count,'files dropped so far')
                continue
    print(dropped_count,'files dropped due to read errors')
    df = pd.DataFrame(data=rows,columns=['url','html','text'])
    return df

def filter_text(df):
    text = df.text.dropna()
    text = text.apply(lambda x: re.sub('\n+','. ',x))
    text = text.apply(lambda x: re.sub('[\r\f\v]','',x))
    text = text.apply(lambda x: re.sub('\t+',' ',x))
    text = text.apply(lambda x: re.sub('[ ]{2,}',' ',x))
    text = text.apply(lambda x: x.encode("ascii", errors="ignore").decode())
    text = text.apply(lambda x: ' '.join(word for word in x.split() if word not in spam))
    relevant = text.apply(lambda x: ('parkland' in x.lower() or \
                                   'marjory stoneman douglas' in x.lower()) and \
                                    'shooting' in x.lower())
    nonempty = text.apply(lambda x: len(x)>0)
    text = text[relevant & nonempty]
    text = pd.DataFrame(data=text,columns=['text'])
    return text

def get_clusters(df):
    df = df.drop_duplicates().reset_index(drop=True)
    nlp = spacy.load('en')
    df['text_processed'] = df.text.apply(lambda x: ' '.join(word for word in simple_preprocess(x)))
    df['tokens'] = df.text_processed.apply(lambda x:' '.join(token.text for token in nlp.tokenizer(x)))
    
    texts = [row.split() for row in df.tokens]
    bigram = Phrases(texts)
    bigram_model = Phraser(bigram)
    texts = [bigram_model[doc] for doc in texts]
    documents = [TaggedDocument(doc,[i]) for i,doc in enumerate(texts)]
    
    doc2vec = Doc2Vec(workers=4,seed=23)
    doc2vec.build_vocab(documents)
    
    for epoch in tqdm(range(10)):
        doc2vec.train(documents,total_examples=doc2vec.corpus_count,epochs=1)
        doc2vec.alpha -= 0.0002
        doc2vec.min_alpha = doc2vec.alpha
        
    X = np.array([doc2vec.infer_vector(text) for text in texts])
    
    model = KMeans(n_clusters=15,n_jobs=-1)
    model.fit(X)
    
    return model, X, texts

def get_repr_docs(model,X,texts,n_docs=1):
    repr_docs = []
    labels = np.array([])
    ranks = np.array([])
    centers = model.cluster_centers_
    yhat = model.predict(X)
    for c in range(model.n_clusters):
        Xc = X[yhat==c]
        dists = np.sqrt(np.sum((Xc-centers[c])**2,axis=1))
        sorts = np.argsort(dists)
        reps = [rep for rep in texts[sorts][:n_docs]]
        repr_docs = repr_docs+reps
        labels = np.concatenate([labels,c*np.ones(n_docs)])
        ranks = np.concatenate([ranks,np.arange(1,n_docs+1)])
    d = {'docs':repr_docs,'labels':labels.astype(int),'rank':ranks.astype(int)}
    repr_df = pd.DataFrame(data=d)
    return repr_df

def get_random_docs(df):
    idx = np.random.permutation(range(len(df)))
    random_repr_docs = df.iloc[idx].text.iloc[:15]
    return random_repr_docs

def seq2seq_summarizer(article_list):
    return []

def pgn_summarizer(article_list):
    return []

def rean_summarizer(article_list):
    return []




    
    
    