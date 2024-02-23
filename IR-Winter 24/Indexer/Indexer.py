import os
import re
from collections import defaultdict
from bs4 import BeautifulSoup
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import sys



class Indexer:
    def __init__(self):
        # Initialize necessary variables
        self.inverted_index = defaultdict(list)
        self.stop_words = set(stopwords.words('english')) # stop words list from nltk
        self.ps = PorterStemmer()
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.max_memory_size = 1e8 # 1 gb as the max size of the in-memory index. made it 0.1gb for now
        self.partial_index_count = 0
        self.output_folder = "partial_indexes"


    def process_document(self, document):
        # Extract information from a document and update the inverted index
        url = document["url"]
        content = document["content"]
        # Process content (remove HTML tags, tokenize, stem, etc.)
        tokens, important_words = self.tokenize_and_stem(content)
        # Update inverted index
        self.update_index(tokens, important_words, url)
        # Check if in-memory index size exceeds threshold, offload to disk if necessary
        if self.memory_size_exceeded():
            print("exceeded")
            self.offload_to_disk()

    def tokenize_and_stem(self, content):
        # Tokenize, remove stop words, and apply stemming
        # You may need to customize this based on your requirements
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text()
        words = re.findall(r'\b([a-zA-Z]{2,})\b', text.lower())  # Tokenization
        
        # Extract important words
        important_words = self.extract_important_words(text)
        
        # Remove stop words and apply stemming
        tokens = [self.ps.stem(word) for word in words if word not in self.stop_words]
        
        return tokens, important_words
    
    # Function to extract important words from headings, titles, or bold text
    def extract_important_words(self, text):
        important_words = re.findall(r'<(h\d|title|b)>(.*?)<\/\1>', text)
        return [word[1] for word in important_words]

    def update_index(self, tokens, important_words, url):
        # Update inverted index with tokens and their occurrences
        for token in set(tokens + important_words):  # Use set to ensure unique tokens in a document
            self.inverted_index[token].append((url, self.calculate_tf_idf(token, tokens, important_words)))

    def calculate_tf_idf(self, token, tokens, important_words):
        # Assign weights to important words
        if token in important_words:
            weight = 2  # Assign a higher weight to important words
        else:
            weight = 1  # Assign a default weight to other words

        # Convert the tokens to a document-like format
        document_text = " ".join(tokens)

        # Fit and transform the document to obtain the TF-IDF matrix
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([document_text])

        # Get the feature names (terms) from the vectorizer
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        # Find the index of the token in the feature names
        token_index = numpy.where(feature_names == token)[0][0] if token in feature_names else -1

        # If the token is found, get its TF-IDF score using the assigned weight, otherwise, set it to 0
        tf_idf_score = tfidf_matrix[0, token_index] * weight if token_index != -1 else 0
        return tf_idf_score
    
    def memory_size_exceeded(self):
        # Check if size of in-memory index exceeds threshold
        #return len(self.inverted_index) > self.max_memory_size
        return sys.getsizeof(self.inverted_index) > self.max_memory_size

    def offload_to_disk(self):
        print("Offloading to disk...")
        print("Index size:", len(self.inverted_index))
        # Serialize in-memory index and write to disk
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)  # Create the output folder if it doesn't exist
        partial_index_path = os.path.join(self.output_folder, f'partial_index_{self.partial_index_count}.json')
        print("Saving to:", partial_index_path)
        with open(partial_index_path, 'w') as index_file:
            json.dump(dict(self.inverted_index), index_file)
        self.partial_index_count += 1
        self.inverted_index = defaultdict(list)
        print("Offload complete.", self.partial_index_count)



    def merge_partial_indexes(self):
        # Merge partial indexes from disk into a single index
        merged_index = defaultdict(list)
        for i in range(self.partial_index_count):
            partial_index_path = os.path.join(self.output_folder, f'partial_index_{i}.json')
            with open(partial_index_path, 'r') as partial_index_file:
                partial_index = json.load(partial_index_file)
                for term, postings in partial_index.items():
                    merged_index[term].extend(postings)
            # os.remove(partial_index_path)  # Remove partial index file after merging
        self.inverted_index = merged_index
    
    def split_index_files(self):
        # Optional: Split merged index into separate files with term ranges
        terms = sorted(self.inverted_index.keys())
        num_files = (len(terms) + 999) // 1000  # Split into files with 1000 terms each
        for i in range(num_files):
            start_index = i * 1000
            end_index = min((i + 1) * 1000, len(terms))
            index_slice = {term: self.inverted_index[term] for term in terms[start_index:end_index]}
            with open(os.path.join(self.output_folder, f'index_{i}.json'), 'w') as index_file:
                json.dump(index_slice, index_file)

    
    def build_index(self, dataset_folder):
        # Iterate through documents in the dataset folder and build the inverted index
        for domain_folder in os.listdir(dataset_folder):
            domain_path = os.path.join(dataset_folder, domain_folder)
            if not os.path.isdir(domain_path):  # Check if the item is a directory
                continue  # Skip directories
            for filename in os.listdir(domain_path):
                file_path = os.path.join(domain_path, filename)
                if not os.path.isfile(file_path):  # Check if the item is a file
                    continue  # Skip non-files
                with open(file_path, 'r', encoding='utf-8') as file:
                    document = json.load(file)
                    self.process_document(document)

        # Merge partial indexes into a single index
        self.merge_partial_indexes()
        self.split_index_files()  # Add this line to call the split_index_files method

        
    def save_index_to_disk(self, output_folder):
        # Save the inverted index to one or more files on disk
        # You may need to customize this based on your requirements
        with open(os.path.join(output_folder, 'inverted_index4.json'), 'w') as index_file:
            json.dump(self.inverted_index, index_file)

# Usage
if __name__ == "__main__":
    indexer = Indexer()
    dataset_folder = "DEV"  # Update with the actual path
    indexer.build_index(dataset_folder)
    output_folder = "inverted-index"  # Update with the desired output path
    indexer.save_index_to_disk(output_folder)
