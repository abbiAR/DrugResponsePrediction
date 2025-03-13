# DrugResponsePrediction

Code related to the study: "Personalised Medicine: Establishing predictive machine learning models for drug responses in patient derived cell culture"

This repository contains code for executing response predictions and the files necessitated.

The four datasets are separated into folders with code and associated data.

GDSC1 and GDSC2 files:
1. script
2. dataset with transformed IC50 values (-1*log10(IC50)).
3. TML completed matrix for use as training data.

PRISM files:
1. script
2. dataset with transformed fold-change values (-1*log2(fold-change))
3. TML completed matrix for use as training data

RX files:
1. script
2. dataset with viability values (normalised against DMSO)
