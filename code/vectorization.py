import spacy
from spacy import util
import numpy as np
from scipy.spatial.distance import euclidean
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import math

# For vectorization, word2vec model, NER, etc
nlp_spacy = spacy.load('pt_core_news_sm')
util.load_model('vectors_spacy', vocab=nlp_spacy.vocab)

def sents_vect(sents):
    sents_as_vector = []
    for s in sents:
        doc = nlp_spacy(s)
        sents_as_vector.append(doc)
    return sents_as_vector

def word_vect(sents):
    vect_words = []
    for s in sents:
        doc = nlp_spacy(s)
        tokens = [token.vector for token in doc]
        vect_words.append(tokens)
    return vect_words

def average_cos_similiraty(text_vectors):
    similarities = []
    for i in range(len(text_vectors)):
        for j in range(i + 1, len(text_vectors)):
            similarity = cosine_similarity([text_vectors[i].vector], [text_vectors[j].vector])[0][0]
            similarities.append(similarity)

    avg_similarity = np.mean(similarities)
    return avg_similarity

def get_orderd_sentences_by_phrase_graph(sentence_vectors, average_cosine_similarity):
    # Calculate cosine similarity between sentences to construct similarity graph
    similarity_graph = np.zeros((len(sentence_vectors), len(sentence_vectors)))
    for i in range(len(sentence_vectors)):
        for j in range(i + 1, len(sentence_vectors)):
            similarity_graph[i, j] = cosine_similarity([sentence_vectors[i].vector], [sentence_vectors[j].vector])[0][0]
            similarity_graph[j, i] = similarity_graph[i, j]
    # Extract traditional graph-based features
    graph = nx.Graph()
    for i in range(len(sentence_vectors)):
        for j in range(i + 1, len(sentence_vectors)):
            similarity_score = similarity_graph[i, j]
            if similarity_score > average_cosine_similarity:
                graph.add_edge(i, j, weight=similarity_score)
    # Extract non-traditional graph-based (heuristic to select most central sentences)
    central_sentences = []
    while graph.number_of_nodes() > 0:
        centrality_scores = nx.degree_centrality(graph)
        central_node = max(centrality_scores, key=centrality_scores.get)
        central_sentences.append(central_node)
        graph.remove_node(central_node)
    ordered_sentences = [sentence_vectors[i] for i in central_sentences]
    return ordered_sentences

def summarize_by_kmeans(sentence_list, original_sentence_list):
   
    raw_sentence_vectors = []
    for s in sentence_list:
        raw_sentence_vectors.append(s.vector)

    sentence_vectors_array = np.array(raw_sentence_vectors)
    max_cluster = len(sentence_vectors_array)
    sum_of_squares = calculate_wcss(sentence_vectors_array, max_cluster)
    n = optimal_number_of_clusters(sum_of_squares, max_cluster)
    # print(len(sentence_vectors_array))
    # print(n)
    #n = 7
    kmeans = KMeans(n_clusters=n, random_state=42)
    cluster_labels = kmeans.fit_predict(sentence_vectors_array)
    most_closest = []
    for i in range(n):
        custom_dict = {}
        for j in range(len(cluster_labels)):
            if cluster_labels[j] == i:
                custom_dict[j] = euclidean(kmeans.cluster_centers_[i], sentence_vectors_array[j])
        most_closest.append(min(custom_dict, key=custom_dict.get))
    
    final_sentence_list = [sentence_list[index] for index in most_closest]
    
    return_stopwords(final_sentence_list, original_sentence_list)
    fs = put_in_order(final_sentence_list, original_sentence_list)
    return fs


def return_stopwords(final_sentence_list, original_sentence_list):
    for sentence in final_sentence_list:
        most_similar_sentence = get_most_similar_sentence(sentence, original_sentence_list).text
        final_sentence_list[final_sentence_list.index(sentence)] = most_similar_sentence
   

        

def put_in_order(final_sentence_list, original_sentence_list):
    fsl_as_vector = [nlp_spacy(s) for s in final_sentence_list]
    order_dict = {}
    for sent in fsl_as_vector:
        most_similar = get_most_similar_sentence(sent, original_sentence_list).text
        order_dict[sent.text] = original_sentence_list.index(most_similar)
    
    return sorted(order_dict, key=order_dict.get)
    


def get_most_similar_sentence(sentence, original_list_sentence):
    orig_sentences = [nlp_spacy(s) for s in original_list_sentence]
    distances = {}
    for s in orig_sentences:
        distance_value = cosine_similarity([sentence.vector], [s.vector])
        distances[s] = distance_value
    
    most = max(distances, key=distances.get)
    return most

def calculate_wcss(data, max_cluster):
    wcss = []
    for n in range(2, max_cluster + 1):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)
    
    return wcss

def optimal_number_of_clusters(wcss, max_cluster):
    x1, y1  = 2, wcss[0]
    x2, y2 = max_cluster, wcss[len(wcss) - 1]
    distances = []
    for i in range(len(wcss)):
        x0 = i + 2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1 )
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    return distances.index(max(distances)) + 2



            
        
        
        

    
    
        
    
