import re
import nltk
from nltk.corpus import stopwords
from nltk.metrics.distance import jaccard_distance, edit_distance
from nltk.util import ngrams
from nltk.stem.snowball import SnowballStemmer
from vectorization import nlp_spacy as portuguese_ner
import os





def tokenize(txt: str):
    """
    Remove non-words from text and split the text into a token list
    After, remove stop words and return the text as tokens
    """
    stop_words = stopwords.words("portuguese")
    #text_without_special_char = re.sub("[0-9\W]", ' ', txt)
    #tokens = nltk.word_tokenize(text_without_special_char, "portuguese")
    # Removing stopwords
    #tokens = [word for word in tokens if word.lower() not in stop_words]
    sent_tokens_tokenized = nltk.sent_tokenize(txt)
    original_sents = nltk.sent_tokenize(txt)
    for i in range(len(sent_tokens_tokenized)):
        # Removing special chars from sentence
        sentence_without_special_char = re.sub("[0-9\W]", ' ', sent_tokens_tokenized[i])
        # Transforms string into separated tokens
        words_list = nltk.word_tokenize(sentence_without_special_char, 'portuguese')
        # Removing stopwords
        words_list = [word for word in words_list if word.lower() not in stop_words]
        # Change the list to be list of sentences, each sentence separeted in tokens
        sent_tokens_tokenized[i] = words_list

    sent_tokens_tokenized = [s for s in sent_tokens_tokenized if s != []]
    return sent_tokens_tokenized, original_sents


def spell_checking(sent_tokens, original_sent_tokens):
    """
    Spell checking using jaccard distance and edit distance
    """
    string_original_sent_tokens = ' '.join(original_sent_tokens)
    with open("../db/vocabulary.txt", "r+", encoding='utf-8') as vocabulary_file:
        correct_words = vocabulary_file.read().splitlines()
        words_to_add = []
        for sentence in sent_tokens:
            for word in sentence:
                new_word = correct_word(correct_words, word, words_to_add)
                if new_word != None and new_word != 'x':
                    sentence[sentence.index(word)] = new_word
                    string_original_sent_tokens = string_original_sent_tokens.replace(word, new_word)
    
        vocabulary_file.writelines(words_to_add)

    return nltk.sent_tokenize(string_original_sent_tokens) # New original tokens replaced with adjusted words. It will be useful when returning stopwords to sentences



def correct_word(correct_words, word, words_to_add):
    # next lines are using the library 'spellchecker' that is much faster but word correction is a little worse
    #spell_checker = SpellChecker(language='pt') # Dicionário português - Portugal
    # corrected_word = spell_checker.correction(word)
    # if corrected_word:
    #     sentence[sentence.index(word)] = corrected_word
    
    lowercase_word = word.lower()
    temp = [(jaccard_distance(set(ngrams(lowercase_word, 2)), set(ngrams(w, 2))), w)
            for w in correct_words if len(w) > 1 and w[0] == lowercase_word[0]]
    closest_word = sorted(temp, key=lambda val: val[0])[0]
    # The word in 'tokens' is the same in the vocabulary file, so no change is needed.
    if closest_word[0] == 0.0:
        return None
    # If there is not the closest word, we will catch all 'close' words ( <= 0.4)
    all_closes = [w for w in temp if round(w[0], 2) <= 0.34 and
                w[1][0] == lowercase_word[0]]
    if all_closes:
        all_edit_distances = [(edit_distance(word, w[1]), w[1]) for w in all_closes]
        closest_word = sorted(all_edit_distances, key=lambda val: val[0])[0][1]
        return closest_word
    else:
       words_to_add.append("\n" + word.lower())
       return 'x'


def ner(sents: list):
    for sentence_tokens in sents:
        text = ' '.join(sentence_tokens)
        document = portuguese_ner(text)
        for named_entity in document.ents:
            try:
                sentence_tokens.remove(str(named_entity))
            except ValueError:
                continue
    


def get_synonymous_list() -> list:
    with open('../db/synonymous.txt', 'r', encoding='latin-1') as file:
        synonymous = file.readlines()
        syn_list = []
        for line in synonymous:
            start_index_syns = line.index('{')
            end_index_syns = line.index('}')
            syns = line[start_index_syns + 1:end_index_syns]
            syn_list.append(syns.split(', '))
        # Used these lines of code to create frequency.txt file if not exists
        if not os.path.exists('../db/frequency.txt'):
            freq_list = []
            for syn in syn_list:
                new_item = [s + '=0' for s in syn]
                freq_list.append(new_item)
            with open('../db/frequency.txt', 'w', encoding='latin-1') as freq_file:
                for freq in freq_list:
                    freq_file.writelines(f'{freq}\n')
            
        return syn_list
        


def get_freq_dict() -> dict:
    with open('../db/frequency.txt', 'r', encoding='latin-1') as file:
        content = file.readlines()
        freq_dict = {}
        for syn_list in content:
            for syn in eval(syn_list):
                syn_info = syn.split('=')
                key = syn_info[0]
                value = int(syn_info[1])
                if not freq_dict.get(key):
                    freq_dict[key] = value
                else:
                    if value > freq_dict[key]:
                        freq_dict[key] = value

    return freq_dict


def higher_freq_syn(syns: list, freq_dict, word: str):
    global_higher_syn = None
    for syn in syns:
        higher_syn = sorted(syn, key=lambda freq: freq_dict[freq], reverse=True)[0]
        if not global_higher_syn:
            global_higher_syn = higher_syn
        else:
            if freq_dict[higher_syn] > freq_dict[global_higher_syn]:
                global_higher_syn = higher_syn
    # if this is True, the global_higher_syn should be the original word
    if freq_dict[global_higher_syn] == freq_dict[word]:
        global_higher_syn = word
    return global_higher_syn


def thesaurus(sents: list, syn_list, freq_dict):
    for sentence in sents:
        for word in sentence:
            word_syns = [s for s in syn_list if word in s]
            if word_syns:
                higher_synonym = higher_freq_syn(word_syns, freq_dict, word)
                freq_dict[word] = freq_dict[word] + 1
                sentence[sentence.index(word)] = higher_synonym


def stemming(sents: list):
    stemmer = SnowballStemmer("portuguese")
    for i in range(len(sents)):
        sents[i] = [stemmer.stem(word) for word in sents[i]]


def convert_to_list_of_sentences(sents: list):
    for i in range(len(sents)):
        new_sentence = ' '.join(sents[i])
        sents[i] = new_sentence

