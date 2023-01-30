# -*- coding: utf-8 -*-
"""
Created on Jan 13, 2021

Versioning:
    
18/02/2021: first version completed
21/11/2022: package completely rewritten and restructured
21/12/2022: add 'ignore' tokens in evaluations
    
@author: compi
"""

import numpy as np
import pandas as pd
from .utils import tokenizer

def levenshtein(x, y, TOKEN=None, **kwargs):
    '''
    Finds the symmetric Levenshtein distance (sum of SUB/INS/DEL) between two strings across a corpus 
    of x-y pairs.    
    The strings are tokenized according to the TOKEN specification (default is string_2_list tokenization).
    x and y position are irrelevant for the measured distance.
    There is no backtracking, and no separate tracking of SUB/INS/DEL 
    
    Note:
    + for token counts 'x' is designed as 'hyp and 'y' as 'ref'
    + for error rate computation: average length of x and y is used in the division
    
    Parameters
    ----------
        x : str or list of str

        y : str or list of str

        TOKEN: None or str ("WORD" or "CHAR")
            see utils.tokenizer()
        
        **kwargs:
            passed to levenshtein_t()
    
    Returns
   --------
        lev_dist:     int
            Levenshtein Distance (combined)
            
        results:      a dictionary with        
                        'hyp_tokens' : (int) number of tokens in hypothesis (x)
                        'ref_tokens' : (int) number of tokens in reference  (y)
                        'total'      : (int) number of edits 
                        'err'        : (float) error rate (in %)    [100*number_of_errors/length_of_reference]
    '''
    
    assert ( type(x)==type(y) ) 
    if isinstance(x,str):
        x = [x]
        y = [y]
        
    total = 0
    hyp_tokens = 0
    ref_tokens = 0
    for xx,yy in zip(x,y):
        xx = tokenizer(xx,TOKEN=TOKEN)
        yy = tokenizer(yy,TOKEN=TOKEN)
        total += levenshtein_t(xx,yy,**kwargs)
        hyp_tokens += len(xx)
        ref_tokens += len(yy)
    err = 200.*total/(hyp_tokens+ref_tokens)
    return total, {'total':total,'err':err,'utterances':len(x),'hyp_tokens':hyp_tokens,'ref_tokens':ref_tokens}

def levenshtein_t(x, y, TOKEN=None):
    '''
    Finds the symmetric Levenshtein distance (sum of SUB/INS/DEL) between two sequences of tokens x and y   
    x and y can be interchanged as Levenshtein distance is symmetric 
    There is no backtracking, and no separate tracking of SUB/INS/DEL 
    
    
    Parameters
    ----------
        x : list
            tokens in list1 (either hypothesis or test)
        y : list
            tokens in list2 (the other)
      
    Returns
   --------
        lev_dist : int
            = sum of Insertions, Substitutions and Deletions 
        
    '''

    # The inputs should be lists of tokens
    assert (type(x) == type(y) == list)

    # The implementation allows for initial insertions and deletions by
    # adding dummy empty token to both sequences reference, implemented by 
    # a. explicitly adding empty token to reference
    # b. initializing first column of the trellis with sequence of deletions  
    y = ['']+y
    Nx = len(x) 
    Ny = len(y) 
    prev = np.zeros(Ny,dtype='int')
    col = np.zeros(Ny,dtype='int')

    # initialization with sequence of deletions
    for j in range(0,Ny):
        col[j] = j

    # recursion over all x's
    for i in range(0, Nx):
        # copying 'current' to 'previous' seems slightly faster than other list switching strategies
        prev = col.copy()
        col[0] = prev[0]+1
        for j in range(1, Ny):
            if x[i] == y[j]: # match
                col[j] = min( prev[j]+1, prev[j-1], col[j-1]+1)
            else: # 
                col[j] = min( prev[j], prev[j-1], col[j-1] ) + 1      
    dist = col[Ny-1]

    return (dist) 


def align(x,y,**kwargs):
    '''
    a tiny wrapper around edit_distance that just returns an alignment of input strings
    
    Parameters:
    -----------
    x,y:   str or list of str
    
    **kwargs:
        see edit_distance()
        
    Returns:
    --------
    alignments
    
    '''
    _, results = edit_distance(x,y,ALIGN=True,**kwargs)
    return(results['align'])


def edit_distance(x, y, TOKEN=None, **kwargs):
    '''
    Find the edit distance between two strings or two sets of string
    
    See edit_distance_t() for details on the **kwargs
    
    Note:
    + for token counts 'x' is designed as 'hyp and 'y' as 'ref'
    
    Parameters
    ----------
        x : str or list of str
            
        y : str or list of str
          
        TOKEN: None or str ("WORD" or "CHAR")
            Tokenization to be applied
            
        **kwargs:
            passed to edit_distance_t()
    
    Returns
   --------
        see edit_distance()
        
    '''
    
    assert ( type(x)==type(y) )
    # handle single utterance also as list
    if isinstance(x,str):
        IS_STR = True
        x = [x]
        y = [y]
    else: IS_STR = False
    
    total = 0
    hyp_tokens = 0
    ref_tokens = 0
    sub_ct = 0
    ins_ct = 0
    del_ct = 0
    edit_dist = 0.
    Alignment = []
    Errors = []
    Edits = []
    comp_ct = 0
    
    for xx,yy in zip(x,y):
        xx = tokenizer(xx,TOKEN=TOKEN)
        yy = tokenizer(yy,TOKEN=TOKEN)
        result1 = edit_distance_t(xx,yy,**kwargs)
        hyp_tokens += result1['hyp_tokens']
        ref_tokens += result1['ref_tokens']
        sub_ct += result1['sub']
        ins_ct += result1['ins']
        del_ct += result1['del']
        edit_dist += result1['edit_dist']
        if 'comp' in result1.keys(): comp_ct +=result1['comp']
        if 'errors' in result1.keys(): Errors.append(result1['errors'])
        if 'edits' in result1.keys(): Edits.append(result1['edits'])
        if 'align' in result1.keys(): Alignment.append(result1['align'])        
        
    total = sub_ct+ins_ct+del_ct

    err = (100.0*total)/ref_tokens  if (ref_tokens != 0) else 0.
    
    results = { 'total':total, 'sub':sub_ct, 'ins':ins_ct, 'del': del_ct,
                'edit_dist':edit_dist,'err':err,
                'hyp_tokens':hyp_tokens,'ref_tokens':ref_tokens,'utterances':len(x)
               }
    if 'comp' in result1.keys(): results['comp'] = comp_ct
    if 'errors' in result1.keys(): results['errors'] = Errors
    if 'edits' in result1.keys(): results['edits'] = Edits
    if 'align' in result1.keys(): results['align'] = Alignment 
    
    # flatten certain output structures for single utterance
    if IS_STR:
        if 'align' in result1.keys(): results['align'] = results['align'][0]
        if 'edits' in result1.keys(): results['edits'] = results['edits'][0]
        if 'errors' in result1.keys(): results['errors'] = results['errors'][0]
        
    return( edit_dist, results)


def edit_distance_t(x=None,y=None,wS=1.1,wI=1.,wD=1.,wC=.2, wG=.99, EPS="_", ALIGN=False, CMPND=[], GB=[], VERBOSE=False):
    '''
    Master routine for edit distance computation between a single hyp(x) and ref(y) pair, 
    where x and y are a list of tokens (tokenized strings).   
    Edit Weights help in making alignments less ambiguous (by nature and code dependent).
    To have a predictable answer and to minimize edit_distance and number_edits at the same time, it is recommended to set: wI,wD < wS < wI+wD
    
    Apart from the standard SUB/INS/DEL, this routine allows for optional Compounding   
    Also, alignments are optionally returned.
  

    Parameters
    ----------
    x : list
        tokens in hypothesis/test
    y : list
        tokens in reference
            
    wS, wI, wD, wC : float, default (1.1, 1.0, 1.0, 0.2)
        edit costs for Substition, Insertion, Deletion and Compound

    wG : float
         relative cost for gobble matching
         
    EPS: str, default = "_"
        Epsilon symbol in alignments
        
    ALIGN:    boolean, default=False
        if True 
            full trellis is maintained, 
            and producing alignment via backtracking
            
    CMPND : list, default = [] 
        list of characters or characterstring that define naive compounding rules
        most common examples or ['','-'] ie. blank and dash compounding
        if not empty the full trellis option needs to be run independent of ALIGN settings

    GB : list, default = []
        list of gobble tokens , i.e. tokens that take sequence of token substitutions
        
    VERBOSE : boolean, default=False
        if True highly VERBOSE printing of internal results (trellis, backtrace, .. )


    Returns
    -------
        a dictionary with
        
        'hyp_tokens' : (int) number of tokens in hypothesis
        'ref_tokens' : (int) number of tokens in reference
        'total'      : (int) number of edits 
        'sub'        : (int) number of substitions 
        'ins'        : (int) number of insertions
        'del'        : (int) number of deletions
        'comp'       : (int) number of compounds (optional)
        'gobble'     : (int) number of errors that could be attributed to ignore tokens (optional)
        'edit_dist'  : (float) weighted edit distance
        'err'        : (float) error rate (in %) 
        'align'      : (dataframe) with alignment (optional)
        'edits'      : (list) sequence of edits (optional)
                
    '''
    
    assert type(x) == type(y) == list
        
    COMPOUNDS = (len(CMPND) > 0) 
    GOBBLE = (len(GB) > 0)
    if(ALIGN | COMPOUNDS | GOBBLE ):
        Edit_Dist, Edit_Counts, Alignment, Edits = _edit_distance_trellis(x,y,wS=wS,wI=wI,wD=wD,wC=wC,wG=wG,EPS=EPS,
            CMPND=CMPND,GB=GB,VERBOSE=VERBOSE)
    else:
        Edit_Dist, Edit_Counts  = _edit_distance_col(x,y,wS=wS,wI=wI,wD=wD,VERBOSE=VERBOSE)  
        
    total = sum(Edit_Counts[0:3]) # don't count compounds in 'total' number of edits
    err = (100.0*total)/len(y) if ( len(y) != 0 ) else 0.
        
    results = {'total':total, 'sub':Edit_Counts[0], 'ins':Edit_Counts[1], 'del':Edit_Counts[2], 'edit_dist':Edit_Dist,'hyp_tokens':len(x),'ref_tokens':len(y),'err':err}
    if(ALIGN): 
        results['align']=Alignment
        results['edits']=Edits
    if(COMPOUNDS): results['comp']=Edit_Counts[3]

    return(results)



def _edit_distance_col(hyp=[],ref=[],wS=1.1,wI=1.,wD=1.,VERBOSE=False):   
    '''
    Weighted Edit Distance computation / matching between two sequences allowing for SUB/INS/DEL
    This is a version with minimal memory consumption O(N), while still returning detailed edit counts 
    
    The memory consumption of this version is O(N) as it only maintains the last 2 columns of the trellis in memory 
    while maintaining partial edit counts for all cells, but not edit sequences.
    Therefore this module is suitable for very long sequences as long as no alignment needs to be returned.
    
    In terms of computation count, this is obviously stilll O(N^2), though sligtly faster than the full trellis version
    _edit_distance() which is O(N^2) for in terms of computation and memory.
    If an alignment is required then the full _edit_distance() should be used.

    Returns
    -------
        Edit_Dist   : float
            Weighted Edit Distance
        Edit_Counts : list of int
            counts of [nsub,nins,ndel]

    '''
    
    # add empty slot at start to allow for initial INS/DEL
    x = ['']+hyp
    y = ['']+ref
    Nx = len(x) 
    Ny = len(y) 

    # initialization
    trellis = np.zeros(Ny)
    edit_ct = np.zeros(Ny,dtype='int')
    edit_counts = np.zeros((Ny,3),dtype='int')
    for j in range(1,Ny):
        trellis[j]= trellis[j-1]+wD        
        edit_ct[j] = j
        edit_counts[j] = edit_counts[j-1] + [0,0,1]
        
    # recursion
    # a full edit-sequence could be obtained by storing a partial edit sequence in each cell, but that would
    # make memory consumption O(N^2)
    for i in range(1, Nx):
        trellis, edits = _column_update(x[i],y,trellis,wS=wS,wI=wI,wD=wD)
        prev_ct = edit_ct.copy()
        prev_counts = edit_counts.copy()
        for j in range(0,Ny):
            if edits[j] == 'H': 
                edit_counts[j] = prev_counts[j-1]
            elif edits[j] == 'S': 
                edit_counts[j] = prev_counts[j-1] + [1,0,0]
            elif edits[j] == 'I': 
                edit_counts[j] = prev_counts[j] + [0,1,0]
            elif edits[j] =='D': 
                edit_counts[j] = edit_counts[j-1] + [0,0,1]
        if VERBOSE:  
            print("col: ",trellis, edits, edit_ct)            
    if VERBOSE: 
        print("Weighted Edit Distance: ",trellis[Ny-1])
        print("Edits (SUB,INS,DEL): ",edit_counts[Ny-1])
    return trellis[Ny-1],edit_counts[Ny-1]

        
def _column_update(xx,y,prev,wS,wI,wD):
    ''' 
    single column update of trellis in weighted edit computation
    This routine applies to elementary weighted edit distance computation allowing for
    Substitutions, Insertions and Deletions

    inputs
        xx     current input
        y      full list of reference
        prev   values for previous column in trellis
        
    returns
        new    np.array of floats
                next column in trellis
        edits  list
                list of applied edits  (one of 'H','S','I','D')
    '''
    Ny = len(y)
    new = prev.copy()
    edits = ['H']*Ny 
    
    # at position '0' only match or insertion is possible
    new[0] = prev[0]+wI
    edits[0] = 'I'

    # recursion over all other positions 1..Ny
    for j in range(1,Ny):
        yy = y[j]
        # 1. match or sub
        jj = j-1
        if (xx==yy):
            new[j] = prev[jj]
            edits[j] = 'H'
        else:
            new[j] = prev[jj] + wS
            edits[j] = 'S'
        # 2. insertion 
        jj = j
        score_ins = prev[jj] + wI
        if score_ins < new[j]:
            new[j] = score_ins
            edits[j] = 'I'
        # 3. deletion                
        jj = j-1
        score_del = new[jj] + wD
        if score_del < new[j]:
            new[j] = score_del
            edits[j] = 'D'
                    
    return new, edits

    
def _edit_distance_trellis(hyp=[],ref=[],wS=1.1,wI=1.,wD=1.,wC=.2, wG=.99, EPS="_", CMPND=[], GB=[], VERBOSE=False):
    '''
    Weighted Edit Distance computation / matching
        + of a single pair of  hyp/ref examples
        + allowing for SUB/INS/DEL
        + allowing for optional Compounding 
    
    This routine computes a full trellis and returns also an alignment of the matched sequences

    
    Parameters
    ----------
    hyp : list 
        tokens in test
    ref : list
        tokens in reference

    wS, wI, wD, wC : float, default (1.1,  1. , 1., .2 ) 
        weights for Substition, Insertion, Deletion and Compound
    
    EPS: str, default = "_"
        Epsilon symbol in alignments
        
    CMPND: list of compound tokens that is allowed
        default [] i.e. NO compounding
        common compouding options are ['','-'] i.e. merging across space or merging with dash

    VERBOSE : boolean, default=False
        if True highly VERBOSE printing of internal results (trellis, backtrace, .. )


    Returns
    -------
        n_Edits   : int
            Number of Edits (SUB+INS+DEL), excluding Compounds
        Edit_Counts      : list of int
            counts of [nsub,nins,ndel,ncomp]
                The last element is only filled if CMPND is not Empty 
                Note: that Compounds contribute to Edit_Dist (alignment criterion),   
                but that these do not contribute to n_Edits(for error rate computations)
        Edit_Dist : float
            Weighted Edit Distance
        Alignment  : DataFrame
            alignment list of tuples [ (x,y) ]
        Edits:
            list of edits

    '''
    x = hyp
    y = ref
    Nx = len(x) 
    Ny = len(y) 
    
    if( Nx == Ny == 0 ): # fetch two empty lists
        if( len(CMPND) > 0 ): return(0.,np.array([0,0,0,0]),[],[])
        else: return(0.,np.array([0,0,0]),[],[])

    trellis = np.zeros((Nx+1, Ny+1),dtype='float32')
    bptr = np.zeros((Nx+1, Ny+1, 2), dtype='int32')
    edits = np.full((Nx+1, Ny+1),"Q",dtype='str')  #'Q' is just a placeholder
    if len(GB) > 0: # gobble weight initialization
        wGS = wG*wS
        wGI = wG*wI
        wGD = wG*wD
        
    for i in range(1,Nx+1):
        trellis[i, 0] = i * wI
        bptr[i,0] = [i-1,0]
        edits[i,0] = 'I'
    for j in range(1,Ny+1):
        trellis[0, j] = j * wD
        bptr[0,j] = [0,j-1]
        edits[0,j] = 'D'

    # forward pass - trellis computation
    # indices i,j apply to the trellis and run from 1 .. N
    # indices ii,jj apply to the data sequence and run from 0 .. N-1
    for i in range(1, Nx+1):
        ii=i-1
        for j in range(1, Ny+1):
            jj=j-1

            # substitution or match
            score_SUB = trellis[i-1,j-1] + int(x[ii]!=y[jj]) * wS
            trellis[i,j] = score_SUB
            bptr[i,j] = [i-1,j-1]         
            if (x[ii]==y[jj]): edits[i,j] = 'H'            
            else: edits[i,j] = 'S'
                
            # insertion and deletions
            score_INS = trellis[i-1,j] + wI            
            if( score_INS < trellis[i,j] ):
                trellis[i,j] = score_INS
                bptr[i,j] = [i-1,j]
                edits[i,j] = 'I'
            score_DEL = trellis[i,j-1] + wD                
            if( score_DEL < trellis[i,j] ):
                trellis[i,j] = score_DEL
                bptr[i,j] = [i,j-1]
                edits[i,j] = 'D'
    
            # gobbling: either y-token should be in the gobble list
            # this is weighted by normal weights and backpointers can be any Levenshtein move
            # hence a gobble token can absorb multiple inputs
            if ( y[jj] in GB ) or ( x[ii] in GB ):
                if(x[ii] == y[jj]): g1 = trellis[i-1,j-1]
                else: gg = trellis[i-1,j-1] + wG
                if (gg <= trellis[i,j]):
                    trellis[i,j] = gg
                    bptr[i,j] = [i-1,j-1]  
                    edits[i,j] = 'G' 
                gg = trellis[i-1,j] + wG
                if (gg <= trellis[i,j]):
                    trellis[i,j] = gg
                    bptr[i,j] = [i-1,j]                  
                    edits[i,j] = 'G' 
                gg = trellis[i,j-1] + wG
                if (gg <= trellis[i,j]):
                    trellis[i,j] = gg
                    bptr[i,j] = [i,j-1]
                    edits[i,j] = 'G'
                    
            # compounds:
            # In first instance compounds are marked as 'x' or 'y' depending on compounding
            # in x or y sequence.   
            # In the alignment output the compounding parts are later rewritten, joined with a '+' in between
            if len(CMPND) > 0:
                # compounds for x
                if( ii>0 ):
                    Cx = False
                    for c_s in CMPND:
                        Cx |= ( (x[ii-1]+ c_s +x[ii]) == y[jj] )
                    score_Cx = trellis[i-2,j-1] + wC
                    if( Cx & (score_Cx < trellis[i,j]) ):
                        trellis[i,j] = score_Cx
                        bptr[i,j] = [i-2,j-1]
                        edits[i,j] = 'x'
                # compounds for y
                if( jj>0 ):
                    Cy = False
                    for c_s in CMPND:
                        Cy |= ( x[ii] == (y[jj-1]+ c_s +y[jj]) )                           
                    score_Cy = trellis[i-1,j-2] + wC
                    if( Cy & (score_Cy < trellis[i,j]) ):
                        trellis[i,j] = score_Cy
                        bptr[i,j] = [i-1,j-2]
                        edits[i,j] = 'y'
            
                    
    # backtracking
    (ix,iy) = bptr[Nx,Ny]
    trace = [ (Nx-1,Ny-1) ]
    while( (ix>0) | (iy>0) ):
        trace.append( (ix-1,iy-1) )
        (ix,iy) = bptr[ix,iy]   
    trace.reverse()
    if (VERBOSE):
        print("Weighted Edit Distance: ", trellis[Nx,Ny])
        print(trellis[0:,0:].T)
        print(edits[0:,0:].T)
        print("Backtrace")        
        print(trace)
        
    # recovering alignment as [ ( x_i, y_j) ] and edit_sequence as [ edit_ij ]
    # the dummy symbol EPS (default='_') is inserted for counterparts of insertions and deletions
    # compounds are merged with '+' between both parts
    alignment = []
    edit_sequence = []
    for k in range(len(trace)):
        (ix,iy) = trace[k]
        edit = edits[ix+1,iy+1]
        if( ix>=0 ): xtoken = x[ix]
        if( iy>=0 ): ytoken = y[iy]
            
        if (edit == 'I'): ytoken = EPS
        elif (edit == 'D'): xtoken = EPS
        elif (edit == 'x'):
            xtoken = x[ix-1] + '+'  + x[ix]
            edit = 'C'
        elif (edit == 'y'):
            ytoken = y[iy-1]+ '+' +y[iy]
            edit = 'C'
            
        edit_sequence.append(edit)
        alignment.append( (xtoken, ytoken))
    
    nins = sum( e == 'I' for e in edit_sequence)
    ndel = sum( e == 'D' for e in edit_sequence)    
    nsub = sum( e == 'S' for e in edit_sequence)
    ncomp = sum( e == 'C' for e in edit_sequence)
      
    if len(CMPND) == 0:
        edit_counts = [nsub, nins, ndel]
    else:
        edit_counts = [nsub, nins, ndel, ncomp]

    
    if (VERBOSE):
        print(alignment)
        print("Number of Tokens (ref): ",Ny)
        print("Edit Counts: Total/Substitutions/Insertions/Deletions: ",(nsub+nins+ndel),nsub,nins,ndel)
        print("Compounds: ", ncomp )

    return(  trellis[Nx , Ny], np.array(edit_counts),alignment,edit_sequence )





