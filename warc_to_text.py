# converts warc file into a pandas dataframe
# each row of dataframe contains url,html,text for a specific html file

from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
from justext import justext, get_stoplist
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import re

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

def no_spam(text):
    spam = ['advertisement', 'getty', 'published', 'javascript', 'updated', 'update',
            'posted', 'read more', 'photo gallery', 'play video', 'caption']
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
            if len(paragraph.text) > 15 and no_spam(paragraph.text):
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
                    text = re.sub('\n+','. ',text)
                    text = re.sub('[\r\f\v]','',text)
                    text = re.sub('\t+',' ',text)
                    text = re.sub('[ ]{2,}',' ', text)
                    rows.append([url,html,text])
            except:
                dropped_count += 1
                print(dropped_count,'files dropped so far')
                continue
    df = pd.DataFrame(data=rows,columns=['url','html','text'])
    return df

if __name__ == '__main__':
    data_path = '/Users/ryankingery/Repos/text-summarization/data/'
    if not Path(data_path).exists():
        Path(data_path).mkdir()
    if not Path(data_path+'html_and_text_big.csv').exists():
        df = process_warc(data_path+'Shooting_Douglas_2018_big.warc')
        df.to_csv(data_path+'html_and_text_big.csv')
    else:
        df = pd.read_csv(data_path+'html_and_text_big.csv',index_col=0)