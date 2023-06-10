from typing import List
from nltk import word_tokenize # the nltk word tokenizer
from spacy.lang.en import English  # for the spacy tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import json, math
from collections import Counter

# gpt2 tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# gpt2 model
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')


def load_corpus(filename: str):
    corpus = None
    with open(filename, "r") as file:
        corpus = [line.strip() for line in file]
    return corpus


def nltk_tokenize(sentence: str):
    nltk_tokenized = None
    nltk_tokenized = word_tokenize(sentence)
    return [] if nltk_tokenized is None else nltk_tokenized # must be type list of strings


def spacy_tokenize(sentence: str):
    spacy_tokenized = None
    nlp = English()
    output = nlp(sentence)
    spacy_tokenized = [token.orth_ for token in output]
    return [] if spacy_tokenized is None else spacy_tokenized # must be type list of strings

def tokenize(sentence: str):
    # wrapper around whichever tokenizer you liked better
    wrapped_output = None
    wrapped_output = nltk_tokenize(sentence)
    return wrapped_output


def count_bigrams(corpus: list):
    bigram_freqs = {}
    for text in corpus:
        tokens = tokenize(text)
        for i in range(len(tokens)):
            if i + 1 > len(tokens) - 1:
                bigram = [tokens[i]]
                bigram=tuple(bigram+['EOS'])
            else:
                bigram = (tokens[i], tokens[i+1])
                
            if bigram in bigram_freqs:
                bigram_freqs[bigram] += 1
            else:
                bigram_freqs[bigram] = 1
    return bigram_freqs


def count_trigrams(corpus: list):
    trigrams_frequencies = {}
    for text in corpus:
        tokens = tokenize(text)
        for i in range(len(tokens)):
            if i + 1 > len(tokens) - 1:
                trigram = [tokens[i]]
                trigram=tuple(trigram+(['EOS']*2))
            elif i + 2 > len(tokens) -1:
                trigram = [tokens[i]]
                trigram=tuple(trigram+['EOS'])
            else:
                trigram = (tokens[i], tokens[i+1], tokens[i+2])
            
            if trigram in trigrams_frequencies:
                trigrams_frequencies[trigram] += 1
            else:
                trigrams_frequencies[trigram] = 1
    return trigrams_frequencies


def bigram_frequency(bigram: str, bigram_frequency_dict: dict):
    frequency_of_bigram = bigram_frequency_dict.get(tuple(bigram.split()), 0)
    return frequency_of_bigram


def trigram_frequency(trigram: str, trigram_frequency_dict: dict):
    frequency_of_trigram = trigram_frequency_dict.get(tuple(trigram.split()), 0)
    return frequency_of_trigram
    

def get_total_frequency(ngram_frequencies: dict):
    total_frequency = 0
    # compute the frequency of all ngrams from dictionary of counts
    total_frequency = sum(ngram_frequencies.values())
    return total_frequency


def get_probability(
        ngram: str,
        ngram_frequencies: dict):
    probability = None
    ngram = tuple(ngram.split())
    ngram_frequency = ngram_frequencies.get(ngram, 0)
    total_frequency = get_total_frequency(ngram_frequencies)
    probability = ngram_frequency / total_frequency
    return probability


def forward_transition_probability(
        seq_of_three_tokens: list,
        trigram_counts: dict,
        bigram_counts: dict
        ):
    bigram_value = 0.0
    trigram_value = 0.0
    fw_prob = 0.0
    bigram = tuple(seq_of_three_tokens[:2])
    trigram = tuple(seq_of_three_tokens)

    bigram_value = bigram_counts.get(bigram, 0)
    trigram_value = trigram_counts.get(trigram, 0)
    fw_prob = trigram_value / bigram_value if bigram_value != 0 else 0

    return fw_prob


def backward_transition_probability(
        seq_of_three_tokens: list,
        trigram_counts: dict,
        bigram_counts: dict
        ):
    bigram_value = 0.0
    trigram_value = 0.0
    bw_prob = 0.0
    bigram = tuple(seq_of_three_tokens[1:])
    trigram = tuple(seq_of_three_tokens)

    bigram_value = bigram_counts.get(bigram, 0)
    trigram_value = trigram_counts.get(trigram, 0)
    bw_prob = bigram_value / trigram_value if trigram_value != 0 else 0
    return bw_prob


def compare_fw_bw_probability(fw_prob: float, bw_prob: float):
    equivalence_test = False
    if fw_prob == bw_prob:
        equivalence_test = True
    return equivalence_test


def sentence_likelihood(
    sentence,  # an arbitrary string
    bigram_counts,   # the output of count_bigrams
    trigram_counts   # the output of count_trigrams
    ):
    likelihood = 0

    tokens = tokenize(sentence)
    for i in range(2, len(tokens)):
        first_token, second_token, third_token = tokens[i-2:i+1]
        
        fw_probability = forward_transition_probability([first_token, second_token, third_token], trigram_counts, bigram_counts)

        if fw_probability > 0:
            log_probability = math.log(fw_probability)
            likelihood += log_probability
    
    return likelihood


def neural_tokenize(sentence: str):
    tokenizer_output = gpt2_tokenizer.encode_plus(
        sentence, return_tensors="pt"
        ) #Encode the text into gpt2 tokens
    return tokenizer_output

def neural_logits(tokenizer_output):
    logits = None
    input_ids = tokenizer_output['input_ids']
    attention_mask = tokenizer_output['attention_mask']
    output = gpt2_model(input_ids, attention_mask=attention_mask)
    logits = output.logits
    return logits

def normalize_probability(logits):
    softmax_logits = None
    sm = torch.nn.Softmax(dim=2)
    softmax_logits =  sm(logits)
    return softmax_logits


def neural_fw_probability(
    softmax_logits,
    tokenizer_output
    ):
    probabilities = []
    input_ids = tokenizer_output['input_ids'][0]

    for i, input_id in enumerate(input_ids):
        token_prob = softmax_logits[:, i, input_id]
        probabilities.append(token_prob)
    return torch.Tensor(probabilities)

def neural_likelihood(diagonal_of_probs):
    likelihood = None
    likelihood = diagonal_of_probs.log().sum()
    return likelihood
