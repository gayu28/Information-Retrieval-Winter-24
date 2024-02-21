import os
import re
from collections import defaultdict
from bs4 import BeautifulSoup
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')

class Indexer:
    def __init__(self):
        # Initialize necessary variables
        self.inverted_index = defaultdict(list)
        self.stop_words = set(stopwords.words('english')) # stop words list from nltk
        self.ps = PorterStemmer()
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')


    def process_document(self, document):
        # Extract information from a document and update the inverted index
        url = document["url"]
        content = document["content"]
        # Process content (remove HTML tags, tokenize, stem, etc.)
        tokens, important_words = self.tokenize_and_stem(content)
        # Update inverted index
        self.update_index(tokens, important_words, url)

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


    def save_index_to_disk(self, output_folder):
        # Save the inverted index to one or more files on disk
        # You may need to customize this based on your requirements
        with open(os.path.join(output_folder, 'inverted_index2.json'), 'w') as index_file:
            json.dump(self.inverted_index, index_file)

# Usage
if __name__ == "__main__":
    indexer = Indexer()
    dataset_folder = "/home/ics-home/IR-Winter 24/Indexer/Data"  # Update with the actual path
    indexer.build_index(dataset_folder)
    output_folder = "inverted-index"  # Update with the desired output path
    indexer.save_index_to_disk(output_folder)
