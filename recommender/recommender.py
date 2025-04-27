from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def recommend(user_interests, paper_abstracts):
    embeddings1 = model.encode(user_interests, convert_to_tensor=True)
    embeddings2 = model.encode(paper_abstracts, convert_to_tensor=True)
    scores = util.cos_sim(embeddings1, embeddings2)
    return scores