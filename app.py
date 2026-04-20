#   IR SYSTEM — search engine over data scraped from www.anpig.it
#
#   Implements: TF-IDF indexing · BM25 scoring · Rocchio relevance feedback
#               Italian stemming/stopword removal via NLTK Snowball

import sys
import nltk
from flask import Flask, render_template, request
import json
import os
import dill as pickle
import math
from collections import defaultdict, Counter
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

INDEX_FILE_PATH = 'my_index.pkl'
DOCUMENTS_FILE_PATH = os.path.join('data', 'dysderadb.anpig_complete.json')


def setup_nltk_resources():
    for resource in ['punkt', 'punkt_tab', 'stopwords']:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource)


def nested_dict():
    return defaultdict(lambda: defaultdict(list))


term_positions = nested_dict()
inverted_index, doc_lengths, term_positions, doc_ids = None, None, None, None
idf_weights = {}


def preprocess(document, language='italian'):
    if not document:
        return [], defaultdict(list)
    stemmer = SnowballStemmer(language)
    stop_words = set(stopwords.words(language))
    tokens = word_tokenize(document.lower(), language=language)
    processed_tokens = []
    positions = defaultdict(list)
    for idx, word in enumerate(tokens):
        if word not in stop_words and word.isalpha():
            stemmed_word = stemmer.stem(word)
            processed_tokens.append(stemmed_word)
            positions[stemmed_word].append(idx)
    return processed_tokens, positions


def create_inverted_index(doc_texts):
    global idf_weights
    inverted_index = defaultdict(set)
    doc_lengths = {}
    term_positions = nested_dict()
    doc_ids = {}
    total_docs = len(doc_texts)

    for doc in doc_texts:
        doc_id = doc["_id"]["$oid"]
        parts = [
            doc.get('name', ''),
            doc.get('titles', ''),
            doc.get('text', ''),
            doc.get('meta', {}).get('description', ''),
            doc.get('meta', {}).get('keywords', '')
        ]
        combined_text = " ".join(filter(None, parts))
        tokens, positions = preprocess(combined_text)
        doc_lengths[doc_id] = len(tokens)
        doc_ids[doc_id] = doc
        for word in tokens:
            inverted_index[word].add(doc_id)
            term_positions[word][doc_id].extend(positions[word])

    idf_weights = {term: math.log((total_docs / len(docs)) + 1) for term, docs in inverted_index.items()}
    return inverted_index, doc_lengths, term_positions, doc_ids


def estimate_term_probabilities(relevant_doc_ids, non_relevant_doc_ids):
    term_relevant_count = Counter()
    term_non_relevant_count = Counter()
    for doc_id in relevant_doc_ids:
        tokens = preprocess(doc_ids[doc_id]['text'])[0]
        term_relevant_count.update(set(tokens))
    for doc_id in non_relevant_doc_ids:
        tokens = preprocess(doc_ids[doc_id]['text'])[0]
        term_non_relevant_count.update(set(tokens))
    total_relevant = len(relevant_doc_ids)
    total_non_relevant = len(non_relevant_doc_ids)
    p_t_given_R = {term: (count / total_relevant) for term, count in term_relevant_count.items()}
    p_t_given_not_R = {term: (count / total_non_relevant) for term, count in term_non_relevant_count.items()}
    return p_t_given_R, p_t_given_not_R


def vectorize_document(tokens, idf_weights):
    vector = defaultdict(float)
    token_counts = Counter(tokens)
    for token, count in token_counts.items():
        vector[token] = (count / len(tokens)) * idf_weights.get(token, 0)
    return vector


def adjust_query_vector(query_vec, p_t_given_R, p_t_given_not_R):
    adjusted_query_vec = defaultdict(float)
    for term, weight in query_vec.items():
        if term in p_t_given_R and term in p_t_given_not_R:
            odds_ratio = p_t_given_R[term] / (p_t_given_not_R[term] + 1)
            adjusted_query_vec[term] = weight * math.log(odds_ratio + 1)
    return adjusted_query_vec


def apply_rocchio(query_vec, relevant_doc_ids, non_relevant_doc_ids, alpha=1.0, beta=0.75, gamma=0.25):
    rel_centroid = defaultdict(float)
    nrel_centroid = defaultdict(float)
    for doc_id in relevant_doc_ids:
        doc_vec = vectorize_document(preprocess(doc_ids[doc_id]['text'])[0], idf_weights)
        for term, weight in doc_vec.items():
            rel_centroid[term] += weight / len(relevant_doc_ids)
    for doc_id in non_relevant_doc_ids:
        doc_vec = vectorize_document(preprocess(doc_ids[doc_id]['text'])[0], idf_weights)
        for term, weight in doc_vec.items():
            nrel_centroid[term] += weight / len(non_relevant_doc_ids)
    new_query_vec = defaultdict(float)
    for term in set(query_vec) | set(rel_centroid) | set(nrel_centroid):
        new_query_vec[term] = (alpha * query_vec.get(term, 0)
                               + beta * rel_centroid.get(term, 0)
                               - gamma * nrel_centroid.get(term, 0))
    return new_query_vec


def bm25(doc_length, avg_doc_length, term_freq, num_docs, doc_freq, k1=1.5, b=0.75):
    idf = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
    tf = ((k1 + 1) * term_freq) / (k1 * (1 - b + b * (doc_length / avg_doc_length)) + term_freq)
    return idf * tf


def search_query(query, inverted_index, doc_ids, doc_lengths, term_positions,
                 relevant_doc_ids=None, non_relevant_doc_ids=None, use_rocchio=True):
    query_tokens = preprocess(query)[0]
    query_vec = vectorize_document(query_tokens, idf_weights)
    if use_rocchio and (relevant_doc_ids or non_relevant_doc_ids):
        query_vec = apply_rocchio(query_vec, relevant_doc_ids, non_relevant_doc_ids)
    scores = defaultdict(float)
    avg_doc_length = sum(doc_lengths.values()) / len(doc_ids)
    for term, weight in query_vec.items():
        if term in inverted_index:
            doc_freq = len(inverted_index[term])
            for doc_id in inverted_index[term]:
                term_freq = len(term_positions[term][doc_id])
                score = bm25(doc_lengths[doc_id], avg_doc_length, term_freq, len(doc_ids), doc_freq)
                scores[doc_id] += weight * score
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(doc_id, score) for doc_id, score in sorted_scores][:10]


def load_index(file_path=INDEX_FILE_PATH):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None, None, None, None


def save_index(index, file_path=INDEX_FILE_PATH):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(index, f)
    except Exception as e:
        print(f"Failed to save index: {e}")


def parse_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_resources():
    global inverted_index, doc_lengths, term_positions, doc_ids, idf_weights
    if not os.path.exists(INDEX_FILE_PATH):
        print("Building index from corpus...")
        documents = parse_json(DOCUMENTS_FILE_PATH)
        if documents:
            inverted_index, doc_lengths, term_positions, doc_ids = create_inverted_index(documents)
            save_index((inverted_index, doc_lengths, term_positions, doc_ids))
            print(f"Index built: {len(doc_ids)} documents, {len(inverted_index)} terms.")
        else:
            print("Document loading failed.")
    else:
        print("Loading existing index...")
        loaded_data = load_index()
        if loaded_data[0] is not None:
            inverted_index, doc_lengths, term_positions, doc_ids = loaded_data
            total_docs = len(doc_ids)
            idf_weights = {term: math.log((total_docs / len(docs)) + 1) for term, docs in inverted_index.items()}
            print(f"Index loaded: {len(doc_ids)} documents, {len(inverted_index)} terms.")


def calculate_evaluation_metrics(results, relevant_doc_ids):
    retrieved_doc_ids = {doc_id for doc_id, _ in results}
    relevant_docs_set = set(relevant_doc_ids)
    tp = len(retrieved_doc_ids & relevant_docs_set)
    fp = len(retrieved_doc_ids - relevant_docs_set)
    fn = len(relevant_docs_set - retrieved_doc_ids)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}


app = Flask(__name__)


@app.route('/')
@app.route('/search')
def search_page():
    return render_template('search.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    results = search_query(query, inverted_index, doc_ids, doc_lengths, term_positions)
    return render_template('results.html', query=query, results=results, doc_ids=doc_ids)


@app.route('/feedback', methods=['POST'])
def feedback():
    query = request.form['query']
    feedback_data = {}
    errors = []
    for key, value in request.form.items():
        if key.startswith('feedback-'):
            doc_id = key.split('-')[1]
            feedback_data[doc_id] = value
    relevant_doc_ids = [doc_id for doc_id, fb in feedback_data.items() if fb == 'relevant']
    non_relevant_doc_ids = [doc_id for doc_id, fb in feedback_data.items() if fb == 'not_relevant']
    query_vec = vectorize_document(preprocess(query)[0], idf_weights)
    p_t_given_R, p_t_given_not_R = estimate_term_probabilities(relevant_doc_ids, non_relevant_doc_ids)
    adjust_query_vector(query_vec, p_t_given_R, p_t_given_not_R)
    updated_results = search_query(query, inverted_index, doc_ids, doc_lengths, term_positions,
                                   relevant_doc_ids, non_relevant_doc_ids, use_rocchio=True)
    metrics_after = calculate_evaluation_metrics(updated_results, relevant_doc_ids)
    if errors:
        return render_template('error.html', error_message=", ".join(errors))
    return render_template('results.html', query=query, results=updated_results,
                           doc_ids=doc_ids, metrics_after=metrics_after)


if __name__ == "__main__":
    setup_nltk_resources()
    load_resources()
    app.run(host='0.0.0.0', port=5002, debug=False)
