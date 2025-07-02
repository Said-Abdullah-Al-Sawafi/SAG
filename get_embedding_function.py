# get_embedding_function.py

from langchain_ollama import OllamaEmbeddings
# from langchain_aws import BedrockEmbeddings # Or langchain_community.embeddings

def get_embedding_function():
    # Comment out the AWS Bedrock line
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )

    # Uncomment the local Ollama line
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings