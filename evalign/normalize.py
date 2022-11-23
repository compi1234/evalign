# -*- coding: utf-8 -*-
"""

@author:  compi
@version: 0.1
@revised: 23/11/2022

"""

import re, sys

#__all__ = ['Lower','Upper']



### Elementary String Operations supported in the Normalizer Class
##################################################################

def Lower(s,arg=None):
    return s.lower()        

def Upper(s,arg=None):
    return s.upper()

def Substitute(s,arg):
    for k,v in arg.items():
        s= s.replace(k,v)
    return s
   
def SubstituteWords(s,arg):
    re_split1 = re.compile(r'(\S+)')
    words = re_split1.split(s)
    # replace item if found in the dictionary, otherwise use itself as default mapping
    s = "".join([arg.get(it, it) for it in words])
    return s

def RemoveTags(s,arg=None):
    return  re.sub(r'<[^>]{0,32}>', r'', s)

def StripHyphen(s,arg=None):
    return re.sub(r' -(\S)', r' \1', re.sub(r'(\S)- ', r'\1 ', s)) 

def SplitHyphen(s,arg=None):
    return re.sub(r'(\S)-(\S)', r'\1 \2',s)

def DecompHyphen(s,arg=None):
    return re.sub(r'-(\S)', r'_\1', re.sub(r'(\S)-', r'\1_', re.sub(r'(\S)-(\S)', r'\1_ _\2', s))) 


def MakeCompounds(s,arg=None): # Assumes single blanks between words
    return  s.replace("_ _", "").replace("_ ", "").replace(" _", "") 

# Remove (multiple) Punctuation Characters if at end of line or followed by white space, i.e. leave them word internal
def RemovePunctuation(s,arg=None):
    s1=re.sub('[\.,:;!?]+$','',s)
    s1=re.sub('[\.,:;!?]+\s',' ',s1)
    return(s1)
    
def RemoveWhiteSpace(s,arg=None):
    return (" ".join(s.split()))   

def SubstituteRegex(s,arg=None):
    for k, v in arg.items():
        s = re.sub(k, v, s)
    return s   

class Normalizer:
    """
    Summary:
    --------
    The Normalizer class delivers a set of text normalization operations
    that are handy in ASR (and NLP) evaluations
    
    A Normalizer object is formed by adding elementary operations 
    to the processing pipeline.
    
    Normalization can then be applied to a string or corpus (list of strings)
    by calling Normalizer.process(text)
    
    The elementary opertions are:
        ToLowerCase          convert to lower case
        ToUpperCase          convert to upper case
        ReduceWhiteSpace    converts all white space in between words to single blank 
        DeleteTags           delete all XML like tagged tokens (i.e. between <>)
        SubstitutePatterns   (recursive) pattern substitution, may include space and punctuation
                                The loop is over the patters, hence substitutions can be applied recursively
        SubstituteWords      (single) word substitutions (word=: anything between white space)
                                The loop is over the words of the input; single substitution per word 
        StripHyphen              removes word initial or final hyphenation character
        SplitHyphen              decompound hyphenated words
        CompoundOnChar           compound words marked with COMPOUND_CHAR
      
    Special Characters:
    -------------------
        HYPHEN_CHAR     : denotes hyphenation, default = '-'
        COMPOUND_CHAR   : denotes optional compounding, default = '_'
        TAG_CHARS       : encloses tags, default = '<>'
        PARSING_CHAR    : parsing character in substitution files
        
        In general it is assumed that everything will be most robust if the defaults
        are preserved for these special characters. In some routines it is possible to redefine these characters.
        
    Attributes:
    -----------
        pipe_names: list, default = []
            list of str with operand name
        pipe_args:
            list with arguments to the operands, typically dict or None
    
    Methods:
    --------
        add_pipe(name,arg):     add operation 'name' with arguments 'arg' to the pipe
        info():                 prints pipeline and arguments
        process(text):          process a text through the pipeline. 'text' is a single string
                                    or list 
    """
    
    def __init__(self):
        self.pipe_names=[]
        self.pipe_args=[]
        
    def add_pipe(self,proc_step,arg=None):
        self.pipe_names.append(proc_step)
        self.pipe_args.append(arg)

    def info(self):
        """
        Print the processing pipeline and its arguments
        """
        
        for name,arg in zip(self.pipe_names,self.pipe_args):
            if arg is None: print(name)
            else: print(name,arg)
            
            
    def process_string(self,s):
        for name,arg in zip(self.pipe_names,self.pipe_args): 
            # call the module defined in this file with the specified name
            proc_module = getattr(sys.modules[__name__],name)
            s = proc_module(s,arg)
        return s
    
    def process(self,text):    
        if isinstance(text, list):
            return [ self.process_string(s) for s in text]
        else:
            return self.process_string(text)   

        

