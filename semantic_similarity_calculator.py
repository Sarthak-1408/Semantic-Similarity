from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSimilarityCalculator:
    def __init__(self, model_name='bert-base-nli-mean-tokens'):
        self.model = SentenceTransformer(model_name)
    
    def calculate_semantic_similarity(self, sentence1, sentence2):
        # Encode the sentences into vectors
        embeddings = self.model.encode([sentence1, sentence2])
        vector1, vector2 = embeddings[0], embeddings[1]

        # Calculate cosine similarity between the vectors
        similarity_score = cosine_similarity([vector1], [vector2])[0][0]

        return similarity_score
