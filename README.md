# ACBD(argument component boundary detection)
Sample code for the article " Joint RNN Model for Argument Component Boundary Detection"

##Data:
    Argument Annotated Essays (version 2)
    From :
    https://www.ukp.tu-darmstadt.de/data/argumentation-mining/argument-annotated-essays-version-2/

##Requirements:
    tensorflow 1.11
    python3 with
        numpy
        gensim
        shutil
        pandas
         ...
    tested on 64-bit Linux versions

##Experiment:
    python main.py --model='rnn'                   |  Bi-LSTM
    python main.py --model='rnn-crf'               |  Bi-LSTM-CRF
    python main.py --mode=1 --model='rnn-crf       |  Bi-LSTM-CRF with knowing sentence's argument status
    python main.py -k=0.3 --model='joint-rnn'       |  Joint-RNN


We have used Google's Tensorflow to implement 3 models in /models. The hyper parameters are present in main.py. Tweaking the parameters can yield a variety of results which are worth noting.

##Data Prepare:
split_data : split the essays into training set(322 essays) and test set(80 essays) as the origin paper(Stab and Gurevych 2016) said.
sentence_token: get the words that consist of a sentence, the sentence's relative location, the argument status of the sentence and each words' BIO tag.

##Get Word Embeddings:
using the Google's word2vec 300 dimensional embeddings trained on google news as the pretrained word embeddings.
