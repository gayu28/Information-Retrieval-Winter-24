import os
import json

def count_documents(dataset_folder):
    total_documents = 0
    for domain_folder in os.listdir(dataset_folder):
        domain_path = os.path.join(dataset_folder, domain_folder)
        if os.path.isdir(domain_path):
            for filename in os.listdir(domain_path):
                total_documents += 1
    return total_documents

def count_unique_tokens(index_folder):
    unique_tokens = set()
    for index_file in os.listdir(index_folder):
        if index_file.endswith('.json'):
            with open(os.path.join(index_folder, index_file), 'r') as file:
                index_data = json.load(file)
                unique_tokens.update(index_data.keys())
    return len(unique_tokens)

def calculate_index_size(index_folder):
    total_size = 0
    for index_file in os.listdir(index_folder):
        if index_file.endswith('.json'):
            total_size += os.path.getsize(os.path.join(index_folder, index_file))
    return total_size / 1024  # Convert bytes to KB

def generate_report(dataset_folder, index_folder, output_file):
    num_documents = count_documents(dataset_folder)
    num_unique_tokens = count_unique_tokens(index_folder)
    index_size_kb = calculate_index_size(index_folder)
    
    with open(output_file, 'w') as report_file:
        report_file.write("Index Statistics\n\n")
        report_file.write("Metric\t\t\tValue\n")
        report_file.write("-------------------------\n")
        report_file.write(f"Number of Documents\t{num_documents}\n")
        report_file.write(f"Number of Unique Tokens\t{num_unique_tokens}\n")
        report_file.write(f"Total Size of Index (KB)\t{index_size_kb:.2f}\n")

# Example usage:
dataset_folder = "/home/ics-home/IR-Winter 24/Indexer/Data"
index_folder = "/home/ics-home/IR-Winter 24/Indexer/inverted-index"
output_file = "index_statistics_report.txt"
generate_report(dataset_folder, index_folder, output_file)
