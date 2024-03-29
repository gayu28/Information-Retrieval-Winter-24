import os
import math
import json
from collections import defaultdict
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.stem import PorterStemmer
import nltk
import logging
import time

class SearchEngine:
    def __init__(self, index_folder):
        self.index_folder = index_folder
        self.ps = PorterStemmer()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.logger = logging.getLogger(__name__)
        self.load_index()

    def load_index(self):
        # Load inverted index from disk
        self.inverted_index = defaultdict(list)
        for filename in os.listdir(self.index_folder):
            if filename.endswith('.json'):
                file_path = os.path.join(self.index_folder, filename)
                with open(file_path, 'r') as f:
                    index_data = json.load(f)
                    for term, postings in index_data.items():
                        self.inverted_index[term] += postings
        
        # Precompute document lengths
        self.doc_lengths = defaultdict(int)
        for postings in self.inverted_index.values():
            for doc_id, tf in postings:
                self.doc_lengths[doc_id] += tf
        
        # Precompute query term weights
        self.query_weights = {}
        N = len(self.inverted_index)
        for term in self.inverted_index.keys():
            df = len(self.inverted_index[term])
            self.query_weights[term] = math.log10(N / df) if df > 0 else 0

    def process_query(self, query):
        # Tokenize and stem the query
        tokens = [self.ps.stem(word) for word in query.split()]
        scores = defaultdict(float)
        N = len(self.inverted_index)  # Total number of documents
        
        # Calculate relevance scores for each document containing any query term
        for token in tokens:
            if token in self.inverted_index:
                df = len(self.inverted_index[token])
                idf = math.log10(N / df) if df > 0 else 0
                for doc_id, tf in self.inverted_index[token]:
                    # TF-IDF calculation
                    tfidf = (1 + math.log10(tf)) * idf
                    
                    # Document length normalization
                    doc_length_norm = self.doc_lengths[doc_id] / max(self.doc_lengths.values())
                    
                    # Query term weighting
                    query_weight = self.query_weights[token]
                    
                    scores[doc_id] += tfidf * doc_length_norm * query_weight

        # Sort documents by their relevance scores
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs

if __name__ == "__main__":
    index_folder = "inverted-index"  # Path to the folder containing the index files
    search_engine = SearchEngine(index_folder)

    while True:
        query = input("Enter your query (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        start_time = time.time()
        results = search_engine.process_query(query)
        end_time = time.time()

        # Output top-ranked documents
        print(f"Search results for '{query}':")
        for i, (doc_id, score) in enumerate(results[:10]):
            print(f"{i+1}. Document ID: {doc_id}, Score: {score}")
        
        print("Time taken:", end_time - start_time, "seconds")
