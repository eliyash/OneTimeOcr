# Mareh OCR - "An One Time OCR"
This project's objective is to allow ocr of any language and font (handwriting also).
The basic idea is instead of relaying on a general algorithm trained on huge datasets,
in this project the training (or transfer learning) 
will be done on the current dataset (i.e. first pages of a book)

## Current state
* a basic tool for marking classifying and viewing the data is ready.
* a basic NN model for detecting letters added (based on EAST word detection network)
* a basic NN model for identifying letters added (simplest vanilla cnn used)
* for gui tkinter was used, for NN pytorch was used

## Todo's by categories (some are optional)
### App
* improve mvc
* add loading data set
* add support for moving / removing letters
* support marking just part of page
* add visualization for training and inference process
* support different boxes sizes
* add duplication detection
* add support for running tessarect
  
### DeepLearning
#### _General_
* chose wisely networks
* add gt page visualization (boxes as image)
#### _Detector net_
* make test data consistent
* ignore misses and false of lettres detection in boundary
#### _Identifier net_
* split letters by logic in train
* add automatic letters clustering
* add support for punctuation (also nikud like in hebrew)
  
### Post processing
* add lettres on page into words and lines tool
* detect spaces
