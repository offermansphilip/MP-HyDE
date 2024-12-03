import numpy as np


class MultiPromptHyDE:
    def __init__(self, promptor1, promptor2, promptor3, promptor4, generator, encoder, searcher):
        self.promptor1 = promptor1
        self.promptor2 = promptor2
        self.promptor3 = promptor3
        self.promptor4 = promptor4
        self.generator = generator
        self.encoder = encoder
        self.searcher = searcher
    
    def prompt(self, query):
        output = []
        output.append(self.promptor1.build_prompt(query))
        output.append(self.promptor2.build_prompt(query))
        output.append(self.promptor3.build_prompt(query))
        output.append(self.promptor4.build_prompt(query))
        return output

    def generate(self, query):
        prompt1 = self.promptor1.build_prompt(query)
        prompt2 = self.promptor2.build_prompt(query)
        prompt3 = self.promptor3.build_prompt(query)
        prompt4 = self.promptor4.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt1, 2)
        hypothesis_documents += self.generator.generate(prompt2, 2) 
        hypothesis_documents += self.generator.generate(prompt3, 2) 
        hypothesis_documents += self.generator.generate(prompt4, 2) 
        return hypothesis_documents
    
    def encode(self, query, hypothesis_documents):
        all_emb_c = []
        for c in [query] + hypothesis_documents:
            c_emb = self.encoder.encode(c)
            all_emb_c.append(np.array(c_emb))
        all_emb_c = np.array(all_emb_c)
        return all_emb_c
    
    def combine(self, vectors):
        avg_emb_c = np.mean(vectors, axis=0)
        hyde_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
        return hyde_vector

    def std(self, vectors):
        # Calculate the standard deviation across all vectors
        embeddings_std_dev = np.std(vectors)
        return embeddings_std_dev
    
    def search(self, hyde_vector, k=10):
        hits = self.searcher.search(hyde_vector, k=k)
        return hits
    
    def e2e_search(self, query, k=10):
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt)
        all_vectors = self.encode(query, hypothesis_documents)
        hyde_vector = self.combine(all_vectors)
        hits = self.search(hyde_vector, k=k)
        return hits