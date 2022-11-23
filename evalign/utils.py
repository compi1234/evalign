# -*- coding: utf-8 -*-
"""
        
@author: compi
"""

import numpy as np
import pandas as pd
from .normalize import RemovePunctuation

def tokenizer(text,TOKEN=None):
    '''
    The tokenizer converts a text to tokens.
    
    Inputs:
    -------
        text    str or list of str
        TOKEN   None, "char" or "word"
            if TOKEN is None then the strings are simply converted to lists without extra processing
            for CHAR or WORD tokens, blank spaces are maximally eliminated
    Ouput:
    ------
        list or list of lists of tokens
    '''
    
    if isinstance(text,list):
        return[ _tokenizer1(s,TOKEN=TOKEN) for s in text ]
    else:
        return _tokenizer1(text,TOKEN=TOKEN)

def _tokenizer1(s,TOKEN=None):
    assert( isinstance(s,str) )
    if TOKEN is None:
        tokens = list(s)
    elif TOKEN.lower() == "word":
        tokens = s.strip().split()
    # convert to character after stripping extraneous white space
    elif TOKEN.lower() == "char": 
        tokens = list(RemovePunctuation(s))
    return(tokens)

def pp_results(results,PRINT='results'):
    '''
    pretty printout of the results dict
    
    you can select one or several of: 'results', 'errors', 'align', 'all'
    '''
    if isinstance(PRINT,list) : selection = PRINT
    elif PRINT == 'results': selection = ['results']
    elif PRINT == 'align': selection = ['align']
    elif PRINT == 'errors': selection = ['errors']
    elif PRINT == 'all': selection = ['results', 'errors','align']
    
    for el in selection:
        if el == 'results':
            print("Error Rate: %.2f%% " % results['err'])
            print("Error Details: #S=%d #I=%d #D=%d" % (results['sub'],results['ins'],results['del']) )
            try: 
                print("Accepted Compounds: #C=%d" % (results['comp']) )
            except: None
            print("Edit Distance: %.2f " % results['edit_dist'])
            print("Tokens (HYP): %d    (REF): %d " % (results['hyp_tokens'], results['ref_tokens']) ) 
            try: print("Utterances: %d" % results['utterances'])   
            except: None

        elif el == 'errors':
            try: 
                print("\nERRORS\n")
                pp_errors(results) 
            except: None
        elif el == 'align':
            try: 
                print("\nALIGNMENTS\n")
                pp_align(results) 
            except: None
                                                                                          

def pp_align(results,indx=None,orient='H'):
    align = results['align']
    edits = results['edits']
    if results['utterances']==1:
        align= [align]
        edits = [edits]
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
        if indx is None:
            for a,e in zip(align,edits):
                _pp_align1(a,e,orient=orient)
        else:
            for i in indx:
                _pp_align1(align[i],edits[i],orient=orient)
                
                
def _pp_align1(a,e,orient='H'):
    df = pd.DataFrame(data=a, index=e, columns=['x','y'])
    if(orient == 'H'): df = df.T
    try:
        display(df)
    except:
        print(df)

def pp_errors(results,orient='V'):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
        df = pd.DataFrame(results['errors'],columns=['x','y','E'])
        if(orient == 'H'): df = df.T
        try:
            display(df)
        except:
            print(df)
            
def pp(res,T=True):
    with pd.option_context('display.max_rows', None, 'display.max_columns', 50): 
        for a,e in zip(res['align'],res['edits']):
            df = pd.DataFrame(data=a, index=e, columns=['x','y'])
            if(T): df = df.T
            display(df)                                                                                          
                                                                            
            
def read_corpus(fname,sentencer='LF',REMOVE_EMPTY_LINES=False,KEYS=False):
    """
    reads a corpus text files and returns corpus as list of utterances
    
    The major options are:
    - defining a dedicated sentencer (default is splitting on line feeds)
    - removing empty lines
    - returning the first words of each line as a KEY
    
    Parameters
    ----------
        fname : str
            file name
        sentencer : str or module
            defines how sentences are cut from the long text, default is by LF
        KEY: boolean
            if True utterance IDs are the first word in an utterance and returned in a separate array
        REMOVE_EMPTY_LINES : boolean
            if True empty lines are removed
    """
    fp = open(fname,"r",encoding="utf-8")
    text = fp.read()
    
    if(sentencer == 'LF'):
        lines = text.splitlines()
    else:
        lines = sentencer(text)
        
    if REMOVE_EMPTY_LINES:
        lines = [ l for l in lines if l ]
        
    if KEYS:
        keys = []
        newlines = []
        for line in lines:
            k,v = line.split(None,1)
            keys.append(k)
            newlines.append(v)
        return(newlines,keys)   
    else:
        return(lines)

def select_from_corpus(src_corpus,src_keys,selection=[]):
    """
    make a KEY based selection from a corpus
    ordering is based on the selection
    """
    assert(len(src_keys) == len(src_corpus))
    
    sel_corpus = []
    sel_keys = []
    corpus_as_dict = dict(zip(src_keys,src_corpus))

    for k in selection:
        if k in src_keys:
            sel_corpus.append(corpus_as_dict[k]) 
            sel_keys.append(k)
        else:
            print("WARNING(select_from_corpus): could not find utt with key: ",k)
    return(sel_corpus, sel_keys)

def match_corpora(corpus1,corpus2,keys1,keys2):
    '''
    Find matching sentences in 2 corpora
    '''
    
    corpus2, keys = select_from_corpus(corpus2, keys2,selection=keys1)
    corpus1, _ = select_from_corpus(corpus1, keys1, selection = keys)
    return(corpus1, corpus2, keys)

def LoadSubstitutionsFromFile(filename,PARSING_CHAR='|'):
    """
    load a substitutions file
    fileformat: lines with
        pattern|replacement|   (assuming PARSING_CHAR=='|)
    returns a dictionary of replacement patterns
    """

    subs = {}
    with open(filename, encoding="utf-8") as f:
        for line in f:
            try:
                (p, s) = line.rstrip().split(PARSING_CHAR, 1)
                subs[p.strip()] = s.rstrip(PARSING_CHAR).strip()
            except ValueError:
                print("WARNING(load_subs_file()): Skipping blanco line in : ", filename)
        return subs


### Text/Corpus Operation Modules
#################################
def CleanCorpus(text,STRIP=True,KEEP_BLANK_LINES=True):
    '''
    splits lines and strips them in a text corpus
    empty lines are removed
    '''
    if isinstance(text,str): text = [text]
    out = []
    for s in text:
        lines = s.splitlines(True)
        print(lines)
        for l in lines:
            if(STRIP): l = l.strip()
            if ((len(l) > 0) | KEEP_BLANK_LINES) : out.append(l)
    return(out)

