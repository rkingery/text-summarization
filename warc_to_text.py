# converts warc file into a pandas dataframe type csv: html_and_text_big.csv
# each row of dataframe contains url,html,text for a specific html file

# also saves a csv of filtered text, containing justext extracted text: text_filtered_big.csv
# stripped of non-ascii and filtered for relevance (Parkland shooting)

# default text extractor is justext (get_text_js)
# can also use beautiful soup for text extractor (get_text_bs)

from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
from justext import justext, get_stoplist
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import re

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

if __name__ == '__main__':
    data_path = Path('/Users/ryankingery/Repos/text-summarization/data/')
    if not data_path.exists():
        data_path.mkdir()
    if not (data_path/'html_and_text_big.csv').exists():
        df = process_warc(data_path/'Shooting_Douglas_big.warc')
        df.to_csv(data_path/'html_and_text_big.csv')
    if not (data_path/'text_filtered_big.csv').exists():
        text = filter_text(df)
        text.to_csv(data_path/'text_filtered_big.csv')
    else:
        df = pd.read_csv(data_path/'html_and_text_big.csv',index_col=0)
        text = pd.read_csv(data_path/'text_filtered_big.csv',index_col=0, header=None)