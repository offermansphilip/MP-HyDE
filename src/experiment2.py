import os
import json
import csv
from tqdm import tqdm
import argparse
import numpy as np
from pyserini.search import FaissSearcher, LuceneSearcher
from pyserini.search.faiss import AutoQueryEncoder
from pyserini.search import get_topics, get_qrels
from sklearn.metrics.pairwise import cosine_similarity

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
    generator070 = OllamaGenerator(model_name=args.model_name, temperature=0.70)

    print("___SINGLE_PROMPTS_HYDE___")
    # Define a list of prompt styles to iterate over
    prompt_styles = ['web search', 'web search novice', 'web search intermediate', 'web search proficient', 'web search expert']

    results = {}  # Initialize a dictionary to store the results

    # Check if the results file exists and load existing results
    output_file = os.path.join(args.run_directory, 'cosine_similarity_results.json')
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)

    # Iterate through prompt styles and calculate cosine similarities
    for i, style1 in enumerate(prompt_styles):
        results.setdefault(style1, {})  # Ensure there's a dict for style1
        for style2 in prompt_styles[i:]:
            results[style1].setdefault(style2, {})  # Ensure there's a dict for style2
            
            promptor1 = Promptor(task=style1)
            hyde1 = HyDE(promptor=promptor1, generator=generator070, encoder=query_encoder, searcher=searcher)

            promptor2 = Promptor(task=style2)
            hyde2 = HyDE(promptor=promptor2, generator=generator070, encoder=query_encoder, searcher=searcher) 

            for qid in tqdm(topics):
                if qid in qrels:
                    query = topics[qid]['title']  # Extract the query text from the topics
                
                    hypothesis_documents1 = hyde1.generate(query)
                    hyde_vectors1 = hyde1.encode(query, hypothesis_documents1)
                    hyde_vector1 = hyde1.combine(hyde_vectors1)

                    hypothesis_documents2 = hyde2.generate(query)
                    hyde_vectors2 = hyde2.encode(query, hypothesis_documents2)
                    hyde_vector2 = hyde2.combine(hyde_vectors2)

                    cosine_similarity_values = cosine_similarity(hyde_vector1, hyde_vector2).flatten().tolist()  # Convert to list for JSON serialization
                    results[style1][style2][qid] = cosine_similarity_values  # Store the result in the dictionary

            # Save results to a JSON file after processing each pair of styles
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)

# Call the main function if this script is run directly
if __name__ == "__main__":
    main()