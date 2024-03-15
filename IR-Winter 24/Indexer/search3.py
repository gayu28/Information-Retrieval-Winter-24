import os
import math
import json
import logging
import time
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix
import networkx as nx

class SearchEngine:
    def __init__(self, index_folder):
        self.index_folder = index_folder
        self.ps = PorterStemmer()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.logger = logging.getLogger(__name__)
        self.load_index()

    def load_index(self):
        # Load inverted index from JSON files
        self.inverted_index = defaultdict(list)
        self.doc_lengths = defaultdict(float)

        # Construct web graph
        self.graph = nx.DiGraph()

        for filename in os.listdir(self.index_folder):
            if filename.endswith('.json'):
                file_path = os.path.join(self.index_folder, filename)
                with open(file_path, 'r') as f:
                    partial_index = json.load(f)
                    for term, postings in partial_index.items():
                        for posting in postings:
                            url, tfidf, positions = posting
                            self.inverted_index[term].append((url, tfidf))  # Append (url, tfidf) tuple
                            self.doc_lengths[url] += tfidf
                            self.graph.add_node(url)
                        self.doc_lengths[url] = math.sqrt(self.doc_lengths[url])  # Precompute document lengths
                        self.graph.add_edges_from([(url, neighbor[0]) for neighbor in postings])

        # Compute PageRank scores
        self.pagerank_scores = nx.pagerank(self.graph)

        # Precompute query term weights
        N = len(self.inverted_index)
        self.query_weights = {term: math.log10(N / len(postings)) if postings else 0
                              for term, postings in self.inverted_index.items()}

    def process_query(self, query):
        # Tokenize and stem the query
        tokens = [self.ps.stem(word) for word in query.split()]
        scores = defaultdict(float)

        # Calculate query vector
        query_vector = np.zeros(len(self.inverted_index))
        for token in tokens:
            if token in self.inverted_index:
                idx = list(self.inverted_index.keys()).index(token)
                query_vector[idx] = self.query_weights[token]

        # Calculate cosine similarity between query vector and document vectors
        for term in tokens:
            if term in self.inverted_index:
                query_weight = self.query_weights[term]
                for url, tfidf in self.inverted_index[term]:
                    scores[url] += query_weight * tfidf

        # Combine relevance scores with PageRank scores
        for url, score in scores.items():
            scores[url] = 0.7 * score + 0.3 * self.pagerank_scores.get(url, 0)

        # Sort documents by their combined scores
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs

if __name__ == "__main__":
    index_folder = "/home/ics-home/IR-Winter 24/Indexer/partial_indexes"  # Path to the folder containing the partial index JSON files
    search_engine = SearchEngine(index_folder)

    while True:
        query = input("Enter your query (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        # Correct spellings of query terms using TextBlob
        corrected_query = query

        start_time = time.time()
        results = search_engine.process_query(corrected_query)
        end_time = time.time()

        # Output top-ranked documents
        print(f"Search results for '{corrected_query}':")
        for i, (url, score) in enumerate(results[:10]):
            print(f"{i+1}. URL: {url}, Score: {score}")

        print("Time taken:", end_time - start_time, "seconds")
