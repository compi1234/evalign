# evalign

A small package for EVALuation and ALIGNing in the context of sequence to sequence matching.

The package includes:
- sequence distance computations
    + Levehnstein
    + Edit_Distance
    + align
    
- text normalization
    + includes a variety of normalization options that can be set as irrelevant in the context of sequence-2-sequence matching
    
- utilities
    + to score (ASR) experiments

## Installation

>  pip install git+https://github.com/compi1234/evalign.git


## Examples

Example notebooks are provided in the test directory:
- eval_test:      tests for routines defined in eval.py module 
- distance_test:  tests for routines defined in the distance.py module
- normalize_test: tests for routines defined in the normalize.py module

If you just want to use it the package for error rate computations on ASR experiments, then just start with **eval_test**;
the other ones are only required if you need more detail
