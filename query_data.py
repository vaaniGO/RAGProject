import argparse
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
google_api_key = os.environ['GOOGLE_API_KEY']


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():

    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    # Using 'models/embedding-001' as a default model. Consider making this configurable
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=google_api_key
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    collection = db._collection  # Access the internal collection
    documents = collection.get()["documents"]  # Retrieve all stored documents
    print(f"Number of documents in the database: {len(documents)}")

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.1:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
    # Use a language model
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
    response = model.invoke(prompt)
    response_text = response.content
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()

