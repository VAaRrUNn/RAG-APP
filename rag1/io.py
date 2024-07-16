import faiss
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

import warnings

warnings.filterwarnings("ignore")


def setup_vector_store_index(filepath):
    print(f"Loading data from {filepath}")
    
    documents = SimpleDirectoryReader(filepath).load_data()

    print("Making vector store and index")

    # By default loading model on CPU
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
        )

    d = 768
    faiss_index = faiss.IndexFlatL2(d)

    vector_store = FaissVectorStore(faiss_index=faiss_index)

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        vector_store=vector_store
    )

    return (index, vector_store)

def get_top_k_matches(query, index, k=2):
    """
    function to retrieve top k matches from vector store
    """

    retriever = index.as_retriever(similarity_top_k=k)
    nodes = retriever.retrieve(query)

    return [node.node.text for node in nodes]

def preprocessing(filepath):

    # Preprocessings
    try:
        index, vector_store = setup_vector_store_index(filepath=filepath)
    except Exception as e:
        print(f"Some error occured during making index and vector store...")
        print("Defaulting to simple chatbot")
        return None, None
    return (index, vector_store)