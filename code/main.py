from nlp import get_freq_dict, get_synonymous_list, tokenize, spell_checking
from nlp import ner, thesaurus, convert_to_list_of_sentences
from vectorization import sents_vect, average_cos_similiraty, get_orderd_sentences_by_phrase_graph, summarize_by_kmeans

if __name__ == "__main__":
    with open("../db/sample3.txt", 'r', encoding='utf-8') as file:
        text_content = file.read()

    # NLP
    syn_list = get_synonymous_list()
    syn_freq_dict = get_freq_dict()
    sents, original_sents = tokenize(text_content)
    ner(sents)
    original_sents = spell_checking(sents, original_sents)
    thesaurus(sents, syn_list, syn_freq_dict)
    # stemming(sents)
    convert_to_list_of_sentences(sents)
    
    # Vetorization
    sent_vector = sents_vect(sents)
    average_cosine_similarity = average_cos_similiraty(sent_vector)
    ordered_sentences = get_orderd_sentences_by_phrase_graph(sent_vector, average_cosine_similarity)
    summary = summarize_by_kmeans(ordered_sentences, original_sents)
    with open('summary.txt', mode='w+', encoding='utf-8') as file:
        for sentence in summary:
            file.write(sentence + '\n\n')  


    
        


