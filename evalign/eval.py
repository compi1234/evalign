# -*- coding: utf-8 -*-
"""
        
@author: compi
"""

import numpy as np
import pandas as pd
from .distance import edit_distance

def eval_corpus(hyp_utt,ref_utt,norm=None,TOKEN="WORD",**kwargs):

    ''' 
    score the output of a speech recognition experiment for WORD or CHARACTER ERROR RATE
    the inputs are matching lists of (or single) sentences
    
    This routine contains following processings steps:
        1. Normalization   according to the *norm* object
        2. Tokenization    to WORDS or CHAR according to TOKENIZER
        3. Scoring         using edit_distance metrics
        
    
    Parameters
    ----------
        hyp_txt : str or list of str
            Hypothesis File Name
        ref_txt : str or list of str
            Reference File Name
        ignore_list : list
            list with KEYs to be ignored
        norm : Normalization object, default is None
            Text Normalization
        TOKEN: None or str, default='WORD'
            should be 'WORD' or 'CHAR' to select WER or CER; 
            when TOKEN is None all white space characters are treated as full valued characters
        **kwargs: 
            keyword arguments to be passed to the edit_distance() scoring routine
            

    Returns
    -------
        results: dict
            Results Dictionary, includes:
                - 'err':  error rate (in %)
                - 'total': total number of edits (SUB+INS+DEL)
                - 'sub','ins', 'del', 'comp'(optional): number of specific edits
                - 'align' (optional) : alignment
                - 'errors': collection of all errors in a dataframe
    '''
   
    # 0. Make sure hyp and ref are lists and of equal length
    if not isinstance(hyp_utt,list): hyp_utt= [ hyp_utt]
    if not isinstance(ref_utt,list): ref_utt= [ ref_utt]
    assert( len(hyp_utt) == len(ref_utt) )
    
    # 1. Normalization ...   
    if norm is not None:
        hyp_utt = [norm.process(utt) for utt in hyp_utt]
        ref_utt = [norm.process(utt) for utt in ref_utt]

    # 2. Tokenization ...
    # 3. Scoring ...
    if TOKEN is not None:
        assert(  (TOKEN=='WORD') | (TOKEN=='CHAR') )  
    _,results = edit_distance(hyp_utt,ref_utt,TOKEN=TOKEN,**kwargs)
    try: # try to extract errors from alignments
        all_errors = [] 
        for al,ed in zip(results['align'],results['edits']):
            utt_errors = [ (a[0],a[1],e) for a,e in zip(al,ed)  if e != 'H']
            all_errors.extend(utt_errors)
        results['errors'] = all_errors 
    except:
        None       
    return(results)
