import numpy as np
import pandas as pd
from pathlib import Path
import os
from utils import *

def main():
    data_path = Path('./data')
    assert((data_path/'Shooting_Douglas_big.warc').exists(),
           'must upload warc file to ./data first')
    
    print('processing warc file')
    df_raw = process_warc(data_path/'Shooting_Douglas_big.warc')
    
    print('filtering text')
    df = filter_text(df_raw)
    
    print('getting doc2vec clusters')
    model, X, texts = get_clusters(df)
    
    print('getting reproducible docs')
    repr_docs_doc2vec = get_repr_docs(model,X,df.text.values).docs.values
    repr_docs_random = get_random_docs(df).values
    
    print('generating and saving summaries')
    summary_path = os.getcwd()+'/summaries/'
    
    summary_list = seq2seq_summarizer(repr_docs_random)
    final_summary = '\n\n'.join(summary_list)
    with open(summary_path+"seq2seq_random.txt", "w") as text_file:
        text_file.write(final_summary)
    
    summary_list = seq2seq_summarizer(repr_docs_doc2vec)
    final_summary = '\n\n'.join(summary_list)
    with open(summary_path+"seq2seq_doc2vec.txt", "w") as text_file:
        text_file.write(final_summary)
    
    summary_list = pgn_summarizer(repr_docs_random)
    final_summary = '\n\n'.join(summary_list)
    with open(summary_path+"pgn_random.txt", "w") as text_file:
        text_file.write(final_summary)
    
    summary_list = pgn_summarizer(repr_docs_doc2vec)
    final_summary = '\n\n'.join(summary_list)
    with open(summary_path+"pgn_doc2vec.txt", "w") as text_file:
        text_file.write(final_summary)
    
    summary_list = rean_summarizer(repr_docs_random)
    final_summary = '\n\n'.join(summary_list)
    with open(summary_path+"rean_random.txt", "w") as text_file:
        text_file.write(final_summary)
    
    summary_list = rean_summarizer(repr_docs_doc2vec)
    final_summary = '\n\n'.join(summary_list)
    with open(summary_path+"rean_doc2vec.txt", "w") as text_file:
        text_file.write(final_summary)
        
    print('done')

if __name__ == '__main__':
    main()