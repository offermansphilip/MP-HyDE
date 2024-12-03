import os
import json
import csv
from tqdm import tqdm
import argparse
import numpy as np
from pyserini.search import FaissSearcher, LuceneSearcher
from pyserini.search.faiss import AutoQueryEncoder
from pyserini.search import get_topics, get_qrels

# Import the classes from your provided module for generating and handling prompts
from hyde import OllamaGenerator, Promptor, HyDE, MultiPromptHyDE
# Import evaluation function
from utils import evaluate_metrics, replace_spaces_with_underscores, create_std_csv

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run MultiPromptHyDE with specified model, encoder, and dataset.")
    
    # Add arguments for configuration
    parser.add_argument('--model_name', type=str, default='llama3.1', help="Name of the text generation model to be used.")
    parser.add_argument('--encoder', type=str, default='facebook/contriever', help="Name of the query encoder model.")
    parser.add_argument('--index_path', type=str, default='./src/contriever_msmarco_index/', help="Path to the Faiss index.")
    parser.add_argument('--prebuilt_index', type=str, default='msmarco-v1-passage', help="Prebuilt Lucene index for passage retrieval.")
    parser.add_argument('--run_directory', type=str, default='./runs/', help="Directory to store the retrieval results.")
    parser.add_argument('--topics_name', type=str, default='dl19-passage', help="Name of the evaluation topic set.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Initialize the query encoder
    query_encoder = AutoQueryEncoder(encoder_dir=args.encoder, pooling='mean')

    # Create a FaissSearcher for dense vector search over the Faiss index
    searcher = FaissSearcher(args.index_path, query_encoder)

    # Load the topics (queries) and the corresponding qrels (ground truth relevance judgments)
    topics = get_topics(args.topics_name)
    qrels = get_qrels(args.topics_name)

    # Initialize the Ollama-based text generators using the specified model and temp
    generator000 = OllamaGenerator(model_name=args.model_name, temperature=0)
    generator035 = OllamaGenerator(model_name=args.model_name, temperature=0.35)
    generator070 = OllamaGenerator(model_name=args.model_name, temperature=0.70)


    print("___CONTRIEVER___")

    trec_file = 'dl19-contriever-top1000-8rep-trec'
    output_csv_file = 'dl19-contriever-top1000-8rep-output.csv'
     # Define the filepaths
    trec_filepath = os.path.join(args.run_directory, trec_file)
    output_csv_filepath = os.path.join(args.run_directory, output_csv_file)

    # Open the output file for writing retrieval results
    with open(trec_filepath, 'w') as f:
        for qid in tqdm(topics):
            if qid in qrels:
                query = topics[qid]['title']  # Extract the query text from the topics

                # Generate Embedding
                # Encode the query using the provided encoder
                query_emb = query_encoder.encode(query)
                # Convert the encoding to a NumPy array
                query_emb = np.array(query_emb)
                # Reshape the embedding to a 2D array with one row
                embedding = query_emb.reshape((1, len(query_emb)))
    
                # Perform a search in the Faiss index using the embedding
                hits = searcher.search(embedding, k=1000)
        
                # Write the top retrieved document results to the output file
                for rank, hit in enumerate(hits, start=1):
                    f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')
                
    # Evaluating metrics
    evaluation_results = evaluate_metrics(trec_filepath)

    # Write the evaluation results to the CSV file
    with open(output_csv_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
    
        # Write header (optional)
        writer.writerow(['Metric', 'Value'])
    
        # Write the rows (each metric and its value)
        writer.writerows(evaluation_results)
    
    print("___SINGLE_PROMPTS_HYDE___")
    # Define a list of prompt styles to iterate over
    prompt_styles = ['web search novice', 'web search intermediate', 'web search proficient', 'web search expert']

    for style in prompt_styles:
        print(f"Running Single Prompt HyDE for prompt style: {style}")
        
        # Initialize a Promptor object for the specified prompt style
        promptor = Promptor(task=style)
        
        # Initialize the HyDE model for generating hypotheses and performing retrieval
        hyde = HyDE(promptor=promptor, generator=generator070, encoder=query_encoder, searcher=searcher)
        
        # Set filenames based on the prompt style
        trec_file = f'single_prompt-hyde-{replace_spaces_with_underscores(style)}-dl19-contriever-llama3.1-0.7-top1000-8rep-trec'
        hypothetical_documents_file = f'single_prompt-hyde-{replace_spaces_with_underscores(style)}-dl19-contriever-llama3.1-0.7-top1000-8rep-hyd.json'
        output_csv_file = f'single_prompt-hyde-{replace_spaces_with_underscores(style)}-dl19-contriever-llama3.1-0.7-top1000-8rep-output.csv'
        std_file = f'single_prompt-hyde-{replace_spaces_with_underscores(style)}-dl19-contriever-llama3.1-0.7-top1000-8rep-std.csv'
        
        # Define the filepaths
        trec_filepath = os.path.join(args.run_directory, trec_file)
        hypothetical_documents_filepath = os.path.join(args.run_directory, hypothetical_documents_file) 
        output_csv_filepath = os.path.join(args.run_directory, output_csv_file)
        std_filepath = os.path.join(args.run_directory, std_file)
        
        stds = []
        # Open the output file for writing retrieval results
        with open(trec_filepath, 'w') as f:
            for qid in tqdm(topics):
                if qid in qrels:
                    query = topics[qid]['title']  # Extract the query text from the topics
                    
                    # Generate hypothesis documents based on the query
                    hypothesis_documents = hyde.generate(query)
                    
                    # Encode the query and hypothesis documents into dense vectors
                    hyde_vectors = hyde.encode(query, hypothesis_documents)

                    hyde_vector = hyde.combine(hyde_vectors)

                    std = hyde.std(hyde_vectors)

                    stds.append((qid, std))
                    
                    # Perform a search in the Faiss index using the generated HyDE vector
                    hits = hyde.search(hyde_vector, k=1000)
                    
                    # Write the top retrieved document results to the output file
                    for rank, hit in enumerate(hits, start=1):
                        f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')
                    
                    # Write the query and hypothesis_documents to the JSON file incrementally
                    with open(hypothetical_documents_filepath, 'a') as hypothetical_documents:
                        json.dump({
                            'query_id': qid,
                            'query': query,
                            'hypothesis_documents': hypothesis_documents
                        }, hypothetical_documents)
                        hypothetical_documents.write('\n')  # Add a newline after each JSON object for separation

        # Evaluate metrics for each prompt style run
        evaluation_results = evaluate_metrics(trec_filepath)

        # Write the evaluation results to the CSV file
        with open(output_csv_filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Metric', 'Value'])  # Write header
            writer.writerows(evaluation_results)  # Write metrics

        create_std_csv(std_filepath, stds)

        print(f"Completed Single Prompt HyDE for prompt style: {style}")
##############################################################################################################

    generators = [generator000, generator035, generator070]

    # Run Single-Prompt HyDE
    print("___SINGLE_PROMPT_HYDE____")
    for generator in generators:
        # Create a Promptor object for generating web search prompts
        promptor = Promptor(task='web search')

        # Initialize the HyDE model for generating hypotheses and performing retrieval
        hyde = HyDE(promptor=promptor, generator=generator, encoder=query_encoder, searcher=searcher)

        temp = generator.get_temperature()
        # Set filenames
        trec_file = f'single_prompt-hyde-web_search-dl19-contriever-llama3.1-{temp}-top1000-8rep-trec'
        hypothetical_documents_file = f'single_prompt-hyde-web_search-dl19-contriever-llama3.1-{temp}-top1000-8rep-hyd.json'
        output_csv_file = f'single_prompt-hyde-web_search-dl19-contriever-llama3.1-{temp}-top1000-8rep-output.csv'
        std_file = f'single_prompt-hyde-web_search-dl19-contriever-llama3.1-{temp}-top1000-8rep-std.csv'

        # Define the filepaths
        trec_filepath = os.path.join(args.run_directory, trec_file)
        hypothetical_documents_filepath = os.path.join(args.run_directory, hypothetical_documents_file) 
        output_csv_filepath = os.path.join(args.run_directory, output_csv_file)
        std_filepath = os.path.join(args.run_directory, std_file)

        stds = []
        # Open the output file for writing retrieval results
        with open(trec_filepath, 'w') as f:
            for qid in tqdm(topics):
                if qid in qrels:
                    query = topics[qid]['title']  # Extract the query text from the topics

                    # Generate hypothesis documents based on the query
                    hypothesis_documents = hyde.generate(query)

                    # Encode the query and hypothesis documents into dense vectors
                    hyde_vectors = hyde.encode(query, hypothesis_documents)

                    hyde_vector = hyde.combine(hyde_vectors)

                    std = hyde.std(hyde_vectors)

                    stds.append((qid, std))

                    # Perform a search in the Faiss index using the generated HyDE vector
                    hits = hyde.search(hyde_vector, k=1000)

                    # Write the top retrieved document results to the output file
                    for rank, hit in enumerate(hits, start=1):
                        f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')
                    
                    # Write the query and hypothesis_documents to the JSON file incrementally
                    with open(hypothetical_documents_filepath, 'a') as hypthetical_documents:
                        json.dump({
                            'query_id': qid,
                            'query': query,
                            'hypothesis_documents': hypothesis_documents
                        }, hypthetical_documents)
                        hypthetical_documents.write('\n')  # Add a newline after each JSON object for separation

        # Example usage for evaluating metrics
        evaluation_results = evaluate_metrics(trec_filepath)

        # Write the evaluation results to the CSV file
        with open(output_csv_filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
        
            # Write header (optional)
            writer.writerow(['Metric', 'Value'])
        
            # Write the rows (each metric and its value)
            writer.writerows(evaluation_results)

        create_std_csv(std_filepath, stds)

    # Run Multi-Prompt HyDE
    print("___MULTI_PROMPT_HYDE____")
    for generator in generators:
        # Create multiple Promptor objects for different perspectives in web search prompts
        promptor1 = Promptor(task='web search')
        promptor2 = Promptor(task='web search expert')
        promptor3 = Promptor(task='web search novice')
        promptor4 = Promptor(task='web search intermediate')

        # Initialize the MultiPromptHyDE model
        hyde = MultiPromptHyDE(promptor1=promptor1, promptor2=promptor2, promptor3=promptor3, promptor4=promptor4, 
                               generator=generator, encoder=query_encoder, searcher=searcher)

        temp = generator.get_temperature()
        # Set filenames
        trec_file = f'multi_prompt-hyde-dl19-contriever-llama3.1-{temp}-top1000-8rep-trec'
        hypothetical_documents_file = f'multi_prompt-hyde-dl19-contriever-llama3.1-{temp}-top1000-8rep-hyd.json'
        output_csv_file = f'multi_prompt-hyde-dl19-contriever-llama3.1-{temp}-top1000-8rep-output.csv'
        std_file = f'multi_prompt-hyde-dl19-contriever-llama3.1-{temp}-top1000-8rep-std.csv'

        # Define the filepaths
        trec_filepath = os.path.join(args.run_directory, trec_file)
        hypothetical_documents_filepath = os.path.join(args.run_directory, hypothetical_documents_file) 
        output_csv_filepath = os.path.join(args.run_directory, output_csv_file)
        std_filepath = os.path.join(args.run_directory, std_file)

        stds = []
        # Open the output file for writing retrieval results
        with open(trec_filepath, 'w') as f:
            for qid in tqdm(topics):
                if qid in qrels:
                    query = topics[qid]['title']  # Extract the query text from the topics

                    # Generate hypothesis documents based on the query
                    hypothesis_documents = hyde.generate(query)

                    # Encode the query and hypothesis documents into dense vectors
                    hyde_vectors = hyde.encode(query, hypothesis_documents)

                    hyde_vector = hyde.combine(hyde_vectors)

                    std = hyde.std(hyde_vectors)

                    stds.append((qid, std))

                    # Perform a search in the Faiss index using the generated HyDE vector
                    hits = hyde.search(hyde_vector, k=1000)

                    # Write the top retrieved document results to the output file
                    for rank, hit in enumerate(hits, start=1):
                        f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')
                    
                    # Write the query and hypothesis_documents to the JSON file incrementally
                    with open(hypothetical_documents_filepath, 'a') as hypthetical_documents:
                        json.dump({
                            'query_id': qid,
                            'query': query,
                            'hypothesis_documents': hypothesis_documents
                        }, hypthetical_documents)
                        hypthetical_documents.write('\n')  # Add a newline after each JSON object for separation

        # Example usage for evaluating metrics
        evaluation_results = evaluate_metrics(trec_filepath)

        # Write the evaluation results to the CSV file
        with open(output_csv_filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
        
            # Write header (optional)
            writer.writerow(['Metric', 'Value'])
        
            # Write the rows (each metric and its value)
            writer.writerows(evaluation_results)

        create_std_csv(std_filepath, stds)

    # Run Multi-Prompt HyDE
    print("___2MULTI_PROMPT_HYDE____")
    for generator in generators:
        # Create multiple Promptor objects for different perspectives in web search prompts
        promptor1 = Promptor(task='web search proficient')
        promptor2 = Promptor(task='web search expert')
        promptor3 = Promptor(task='web search novice')
        promptor4 = Promptor(task='web search intermediate')

        # Initialize the MultiPromptHyDE model
        hyde = MultiPromptHyDE(promptor1=promptor1, promptor2=promptor2, promptor3=promptor3, promptor4=promptor4, 
                               generator=generator, encoder=query_encoder, searcher=searcher)

        temp = generator.get_temperature()
        # Set filenames
        trec_file = f'2multi_prompt-hyde-dl19-contriever-llama3.1-{temp}-top1000-8rep-trec'
        hypothetical_documents_file = f'2multi_prompt-hyde-dl19-contriever-llama3.1-{temp}-top1000-8rep-hyd.json'
        output_csv_file = f'2multi_prompt-hyde-dl19-contriever-llama3.1-{temp}-top1000-8rep-output.csv'
        std_file = f'2multi_prompt-hyde-dl19-contriever-llama3.1-{temp}-top1000-8rep-std.csv'

        # Define the filepaths
        trec_filepath = os.path.join(args.run_directory, trec_file)
        hypothetical_documents_filepath = os.path.join(args.run_directory, hypothetical_documents_file) 
        output_csv_filepath = os.path.join(args.run_directory, output_csv_file)
        std_filepath = os.path.join(args.run_directory, std_file)

        stds = []
        # Open the output file for writing retrieval results
        with open(trec_filepath, 'w') as f:
            for qid in tqdm(topics):
                if qid in qrels:
                    query = topics[qid]['title']  # Extract the query text from the topics

                    # Generate hypothesis documents based on the query
                    hypothesis_documents = hyde.generate(query)

                    # Encode the query and hypothesis documents into dense vectors
                    hyde_vectors = hyde.encode(query, hypothesis_documents)

                    hyde_vector = hyde.combine(hyde_vectors)

                    std = hyde.std(hyde_vectors)

                    stds.append((qid, std))

                    # Perform a search in the Faiss index using the generated HyDE vector
                    hits = hyde.search(hyde_vector, k=1000)

                    # Write the top retrieved document results to the output file
                    for rank, hit in enumerate(hits, start=1):
                        f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')
                    
                    # Write the query and hypothesis_documents to the JSON file incrementally
                    with open(hypothetical_documents_filepath, 'a') as hypthetical_documents:
                        json.dump({
                            'query_id': qid,
                            'query': query,
                            'hypothesis_documents': hypothesis_documents
                        }, hypthetical_documents)
                        hypthetical_documents.write('\n')  # Add a newline after each JSON object for separation

        # Example usage for evaluating metrics
        evaluation_results = evaluate_metrics(trec_filepath)

        # Write the evaluation results to the CSV file
        with open(output_csv_filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
        
            # Write header (optional)
            writer.writerow(['Metric', 'Value'])
        
            # Write the rows (each metric and its value)
            writer.writerows(evaluation_results)
        
        create_std_csv(std_filepath, stds)

if __name__ == "__main__":
    main()
