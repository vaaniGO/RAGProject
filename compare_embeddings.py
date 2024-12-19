import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
google_api_key = os.environ['GOOGLE_API_KEY']

def cosine_similarity(vec1, vec2):
    """Computes the cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def main():
    # Get embedding for a word.
   # Assuming you want to use 'models/embedding-001' 
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=google_api_key
    )
    
    vector_apple = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector_apple}")
    print(f"Vector length: {len(vector_apple)}")

    # Compare vector of two words
    words = ("apple", "iphone")
    vector_iphone = embedding_function.embed_query("iphone")
    similarity = cosine_similarity(vector_apple, vector_iphone)
    print(f"Comparing ({words[0]}, {words[1]}): {similarity}")
    

if __name__ == "__main__":
    main()

    