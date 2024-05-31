#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Students:
#     - Guerrouache Hiba
#     - Djeblahi Maria

"""QUESTIONS/ANSWERS

----------------------------------------------------------
Q1: Why the two sentences "cats chase mice" and "mice chase cats" are considered similar using all words and sentences encoding?
	Propose a solution to fix this.

A: Centroid encoding computes the average word vector of a sentence, disregarding word order. Since both sentences contain the same words ("cats," "chase," "mice"), their centroids will be identical. This method captures the general topic of the sentences but fails to differentiate their distinct meanings due to the lack of consideration for word order.

Proposed Solution:

To address this limitation, incorporating encoding methods that preserve word order is essential. Two potential solutions include:

Recurrent Neural Networks (RNNs): RNNs process words sequentially, capturing the order information crucial for distinguishing between "cats chase mice" and "mice chase cats."
Transformers and Attention Mechanisms: These models consider both word sequence and contextual importance, providing a more nuanced understanding of sentence structure and semantics.
----------------------------------------------------------

----------------------------------------------------------
Q2:  Why using concatenation, the 2nd and 4th sentences are similar?
	Can we enhance this method, so they will not be as similar?

A2: because this method involves  truncating sentences to a fixed length and then concatenating their word vectors. If two sentences share many of the same words and their word order within the fixed length is similar, 
their concatenated vectors will also be similar.

Solution:
Enhance the method by:

Adjust Concatenation Length: Instead of truncating when the sentence length exceeds a predefined limit, consider resizing the vectors to ensure all words are included.
Contextual Embeddings: Use embeddings like BERT or GPT that capture richer contextual information, reducing the chance of different sentences being represented similarly just because they share some words.---------------------------------------------------------

----------------------------------------------------------
Q3: compare between the two sentence representation, indicating their limits.

A3: 
Centroid Encoding:

Pros: Simple and efficient, capturing the overall semantic field of a sentence, Reduces dimensionality compared to bag-of-words.
Cons: Ignores word order, potentially leading to misinterpretations of sentence meaning.

Concatenation of Word Vectors:

Pros: Retains specific word information and potential nuances , Preserves word order within the fixed length. , Captures more detailed sentence structure compared to averaging
Cons: High dimensionality, lack of inherent consideration for word order , May still fail to capture long-range dependencies in longer sentences , ----------------------------------------------------------

"""

import re
import os
import sys
import math
import json
import random
from functools import reduce
from typing import Dict, List, Tuple, Set, Union, Any

# ====================================================
# ============== Usefull functions ===================
# ====================================================

def vec_plus(X: List[float], Y: List[float]) -> List[float]:
    """Given two lists, we calculate the vector sum between them

    Args:
        X (List[float]): The first vector
        Y (List[float]): The second vector

    Returns:
        List[float]: The vector sum (element-wize sum)
    """
    return list(map(sum, zip(X, Y)))

def vec_divs(X: List[float], s: float) -> List[float]:
    """Given a list and a scalar, it returns another list
       where the elements are divided by the scalar

    Args:
        X (List[float]): The list to be divided
        s (float): The scalar

    Returns:
        List[float]: The resulted div list.
    """
    return list(map(lambda e: e/s, X))

# ====================================================
# ====== API: Application Programming Interface ======
# ====================================================

class WordRep:

    def fit(self, text: List[List[str]]) -> None:
        raise 'Not implemented, must be overriden'

    def transform(self, words: List[str]) -> List[List[float]]:
        raise 'Not implemented, must be overriden'

class SentRep:
    def __init__(self, word_rep: WordRep) -> None:
        self.word_rep = word_rep

    def transform(self, text: List[List[str]]) -> List[List[float]]:
        raise 'Not implemented, must be overriden'

class Sim:
    def __init__(self, sent_rep: SentRep) -> None:
        self.sent_rep = sent_rep

    def calculate(self, s1: List[str], s2: List[str]) -> float:
        raise 'Not implemented, must be overriden'
    

# ====================================================
# =========== WordRep implementations ================
# ====================================================

class OneHot(WordRep):
      
        #complete this
    #redifine the fit and transform 
    # vocab begin with unk , special token must be after the unk
    # the token are added by their order of appearance
    # specials tokens are added into specials 
    
    def __init__(self, specials: List[str] = []) -> None:
        super().__init__()
        self.specials = ['[UNK]']+specials
        self.vocab = {}
        
        self.vocab_size = 0

    def fit(self, text: List[List[str]]) -> None:
        self.vocab = {}
        idx = 0
      

        # Add special tokens
        for token in self.specials:
            if token != ['[UNK]']:  # Avoid adding 'unk' again if it is in specials
                self.vocab[token] = idx
                idx += 1
      
        
        # Add unique words from the text
        for sentence in text:
            for word in sentence:
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1

        self.vocab_size = len(self.vocab)
    
    def transform(self, words: List[str]) -> List[List[float]]:
        one_hot_vectors = []
        for word in words:
            vector = [0] * self.vocab_size
            if word in self.vocab:
                vector[self.vocab[word]] = 1
            else:
                # Handle unknown words
                vector[self.vocab['[UNK]']] = 1
            one_hot_vectors.append(vector)
        return one_hot_vectors
     
        
class TermTerm(WordRep):
    def __init__(self, window=2) -> None:
        super().__init__()
        self.window = window
        self.vocab = {}
        self.vocab_size = 0
        self.co_occurrence_matrix = None

    def fit(self, text: List[List[str]]) -> None:
        self.vocab = {}
        idx = 0  # Start index at 0

        # Add unique words from the text
        for sentence in text:
            for word in sentence:
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1

        self.vocab_size = len(self.vocab)
        
        # Initialize co-occurrence matrix
        self.co_occurrence_matrix = [[0] * self.vocab_size for _ in range(self.vocab_size)]
        
        # Compute co-occurrence matrix
        for sentence in text:
            sentence_idx = [self.vocab[word] for word in sentence if word in self.vocab]
            for i in range(len(sentence_idx)):
                for j in range(max(0, i - self.window), min(len(sentence_idx), i + self.window + 1)):
                    if i != j:
                        self.co_occurrence_matrix[sentence_idx[i]][sentence_idx[j]] += 1

    def transform(self, words: List[str]) -> List[List[int]]:
        word_vectors = []
        for word in words:
            if word in self.vocab:
                word_index = self.vocab[word]
                word_vector = self.co_occurrence_matrix[word_index]
                word_vectors.append(word_vector)
            else:
                # If the word is not in the vocab, return a zero vector
                word_vectors.append([0] * self.vocab_size)
        return word_vectors
    
# ====================================================
# =========== SentRep implementations =================
# ====================================================

class ConcatSentRep(SentRep):
    def __init__(self, word_rep: WordRep, max_words: int = 10) -> None:
        super().__init__(word_rep)
        self.max_words = max_words

    def transform(self, text: List[List[str]]) -> List[List[float]]:
        sent_representations = []
        for sentence in text:
            # Get word representations for each word in the sentence
            word_representations = self.word_rep.transform(sentence)
            # Pad or truncate word representations to match max_words
            if len(word_representations) < self.max_words:
                # Pad with zeros if the sentence is shorter than max_words
                pad_length = self.max_words - len(word_representations)
                word_representations.extend([[0.0] * len(word_representations[0])] * pad_length)
            elif len(word_representations) > self.max_words:
                # Truncate if the sentence is longer than max_words
                word_representations = word_representations[:self.max_words]
            # Flatten the list of word representations and append to sentence representations
            sent_representation = [item for sublist in word_representations for item in sublist]
            sent_representations.append(sent_representation)
        return sent_representations

class CentroidSentRep(SentRep):
    def transform(self, text: List[List[str]]) -> List[List[float]]:
        sent_rep = []
        for sentence in text:
            word_vectors = self.word_rep.transform(sentence)
            if word_vectors:
                # Calculate the centroid by summing the word vectors
                centroid = reduce(vec_plus, word_vectors)
                # Divide the centroid vector by the number of word vectors to get the average
                centroid = vec_divs(centroid, len(word_vectors))
            else:
                # If no word vectors are available, use zero vectors
                centroid = [0] * len(word_vectors[0])
            sent_rep.append(centroid)
        return sent_rep

# ====================================================
# =========== Sim implementations ================
# ====================================================

class EuclideanSim(Sim):
    def calculate(self, centroid1: List[float], centroid2: List[float]) -> float:
        # Calculate the squared difference between centroids
        squared_diff = [(c1 - c2) ** 2 for c1, c2 in zip(centroid1, centroid2)]
        
        # Sum the squared differences
        sum_squared_diff = sum(squared_diff)
        
        # Calculate the Euclidean distance
        euclidean_dist = math.sqrt(sum_squared_diff)
        
        # Calculate the similarity using the given equation
        similarity = 1 / (1 + euclidean_dist)
        
        return similarity




# # ====================================================
# # ============ SentenceComparator class ==============
# # ====================================================
# NOT important

# ====================================================
# ===================== Tests ========================
# ====================================================

train_data = [
    ['a', 'computer', 'can', 'help', 'you'],
    ['he', 'can', 'help', 'you', 'and', 'he', 'wants', 'to', 'help', 'you'],
    ['he', 'wants', 'a', 'computer', 'and', 'a', 'computer', 'for', 'you']
]

test_data_wd = ['[S]', 'a', 'computer', 'wants', 'to', 'be', 'you']

test_data_st = [
    ['a', 'computer', 'can', 'help'],
    ['a', 'computer', 'can', 'help', 'you'],
    ['you', 'can', 'help', 'a', 'computer'],
    ['a', 'computer', 'can', 'help', 'you', 'and', 'you', 'can', 'help']
]

class DummyWordRep(WordRep):
    def __init__(self) -> None:
        super().__init__()
        self.code = {
            'a':        [1., 0., 0.],
            'computer': [0., 1., 0.],
            'can':      [0., 0., 1.],
            'help':     [1., 1., 0.],
            'you' :     [1., 0., 1.],
            'he':       [0., 1., 1.],
            'and':      [2., 0., 0.],
            'wants':    [0., 2., 0.],
            'to':       [0., 0., 2.],
            'for':      [2., 1., 0.]
        }

    def transform(self, words: List[str]) -> List[List[float]]:
        res = []
        for word in words:
            res.append(self.code[word])
        return res

def test_OneHot():
    oh_ref  = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    ohs_ref = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    
    oh_enc = OneHot()
    ohs_enc = OneHot(specials=['[CLS]', '[S]'])

    oh_enc.fit(train_data)
    ohs_enc.fit(train_data)

    print('=========================================')
    print('OneHot class test')
    print('=========================================')
    oh_res = oh_enc.transform(test_data_wd)
    print('oneHot without specials')
    print(oh_res)
    print('should be')
    print(oh_ref)

    ohs_res = ohs_enc.transform(test_data_wd)
    print('oneHot with specials')
    print(ohs_res)
    print('should be')
    print(ohs_ref)

def test_TermTerm():
    tt2_ref = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 4, 1, 0, 0, 1, 2, 1, 0, 1], 
               [4, 0, 1, 1, 1, 0, 2, 1, 0, 1], 
               [1, 1, 0, 1, 0, 2, 1, 0, 1, 0], 
               [0, 0, 0, 1, 1, 1, 0, 1, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 1, 2, 3, 0, 1, 1, 0, 1, 1]]
    tt3_ref = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [2, 4, 1, 1, 1, 1, 2, 1, 0, 1], 
               [4, 2, 1, 1, 2, 1, 2, 1, 0, 1], 
               [1, 1, 0, 1, 2, 2, 2, 0, 1, 0], 
               [0, 0, 0, 1, 1, 1, 1, 1, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [1, 2, 2, 3, 0, 2, 1, 2, 1, 1]]

    tt2_enc = TermTerm()
    tt3_enc = TermTerm(window=3)

    tt2_enc.fit(train_data)
    tt3_enc.fit(train_data)

    print('=========================================')
    print('TermTerm class test')
    print('=========================================')
    tt2_res = tt2_enc.transform(test_data_wd)
    print('TermTerm window=2')
    print(tt2_res)
    print('should be')
    print(tt2_ref)

    tt3_res = tt3_enc.transform(test_data_wd)
    print('TermTerm window=3')
    print(tt3_res)
    print('should be')
    print(tt3_ref)

def test_ConcatSentRep():
    test = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0], 
            [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]]
    dummy_wd = DummyWordRep()
    concat_sent = ConcatSentRep(dummy_wd, max_words=5)
    print('=========================================')
    print('ConcatSentRep class test')
    print('=========================================')

    print(concat_sent.transform(test_data_st))
    print('must be')
    print(test)


def test_CentroidSentRep():
    test = [[0.5, 0.5, 0.25], 
            [0.6, 0.4, 0.4], 
            [0.6, 0.4, 0.4], 
            [0.7777777777777778, 0.3333333333333333, 0.4444444444444444]]
    dummy_wd = DummyWordRep()
    centroid_sent = CentroidSentRep(dummy_wd)
    print('=========================================')
    print('CentroidSentRep class test')
    print('=========================================')

    print(centroid_sent.transform(test_data_st))
    print('must be')
    print(test)

def test_Sim():
    # Expected results are now calculated based on the Euclidean distance
    expected_similarities = [
        (0, 1, 1 / (1 + math.sqrt(sum([(0.6 - 0.5) ** 2, (0.4 - 0.5) ** 2, (0.4 - 0.25) ** 2])))),
        (1, 2, 1 / (1 + math.sqrt(sum([(0.6 - 0.6) ** 2, (0.4 - 0.4) ** 2, (0.4 - 0.4) ** 2])))),
        (0, 3, 1 / (1 + math.sqrt(sum([(0.7778 - 0.5) ** 2, (0.3333 - 0.5) ** 2, (0.4444 - 0.25) ** 2])))),
        (2, 3, 1 / (1 + math.sqrt(sum([(0.7778 - 0.6) ** 2, (0.3333 - 0.4) ** 2, (0.4444 - 0.4) ** 2]))))
    ]

    dummy_wd = DummyWordRep()
    centroid_sent = CentroidSentRep(dummy_wd)
    sim = EuclideanSim(centroid_sent)
    
    # Transform the test data sentences to their centroids
    centroids = centroid_sent.transform(test_data_st)

    print('=========================================')
    print('EuclideanSim class test')
    print('=========================================')

    for idx1, idx2, expected_similarity in expected_similarities:
        calculated_similarity = sim.calculate(centroids[idx1], centroids[idx2])
        print(f"Similarity between sentence {idx1+1} and {idx2+1}: {calculated_similarity:.5f} (expected: {expected_similarity:.5f})")



# TODO: activate one test at once 
if __name__ == '__main__':
    test_OneHot()
    #test_TermTerm()
    #test_ConcatSentRep()
    #test_CentroidSentRep()
    test_Sim()
    