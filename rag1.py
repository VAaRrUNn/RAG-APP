import faiss
from transformers import pipeline
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import gradio as gr

import argparse
from functools import partial
import sys
import warnings

warnings.filterwarnings("ignore")


SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer."""

def load_model(checkpoint = None):

    if checkpoint is None:
        checkpoint = "microsoft/Phi-3-mini-4k-instruct"

    pipe = pipeline("text-generation", model=checkpoint, trust_remote_code=True)
    return pipe

def test_model(pipe):
    """simple test function to test the working of model"""
    try:
        query = "who are you and what can you do..."
        output, res = generate(pipe, query)
        print(f"Testing... model")
        print(f"Query: {query}")
        print(f"Output: {res}")
        print(f"Success...")
    except Exception as e:
        print("Error while processing model...")
        sys.exit(0)


def setup_vector_store_index(filepath):
    print("Loading data")
    documents = SimpleDirectoryReader(filepath).load_data()

    print("Making vector store and index")
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2")

    d = 768
    faiss_index = faiss.IndexFlatL2(d)

    vector_store = FaissVectorStore(faiss_index=faiss_index)

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        vector_store=vector_store
    )
    return (index, vector_store)


def get_top_k_matches(query, index, k=3):
    """
    function to retrieve top k matches from vector store
    """
    retriever = index.as_retriever(similarity_top_k=k)
    nodes = retriever.retrieve(query)
    return [node.node.text for node in nodes]


def format_prompt(prompt, retrieved_docs):
    PROMPT = f"Question:{prompt}\nContext:"
    for text in retrieved_docs:
        PROMPT += f"{text}."
    return PROMPT


# Generate function
def generate(pipe, 
             formatted_prompt,
             max_new_tokens=100,
             temperature=0.7,
             top_p=0.9,
             do_sample=True,
             repetition_penalty=1.1):

    formatted_prompt = formatted_prompt[:2000]  # to avoid OOM
    messages = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": formatted_prompt}]

    output = pipe(
        messages,
        max_new_tokens=max_new_tokens,  # Increase this value for longer outputs
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )

    # getting the last response
    response = output[0]["generated_text"][-1]["content"]
    return output, response

def preprocessing(filepath):

    # Preprocessing and loading
    index, vector_store = setup_vector_store_index(filepath=filepath)
    pipe = load_model()

    return (pipe, index, vector_store)


def gen(pipe,
        index,
        query):
    # get all related docs
    retrieved_docs = get_top_k_matches(query, index) # list of string

    # generate output
    formatted_prompt = format_prompt(query, retrieved_docs)

    output, response = generate(pipe, formatted_prompt)
    return response


## UI (Streamlit)

def generate_response(statics, message):
    res = gen(**statics,
              query=message)
    return res


def _chat(statics, message, history):
    response = generate_response(statics, message)
    return response


flag = 0


def _main():
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Simple CLI to take a string input")
    parser.add_argument('-f', '--filepath', type=str,
                        default="../content/data", help="file dir for PDF/txt.. files")

    args = parser.parse_args()
    filepath = args.filepath

    pipe = index = vector_store = None

    global flag
    if flag == 0:
        pipe, index, vector_store = preprocessing(
            filepath=filepath)
        test_model(pipe)
        flag = 1

    statics = {
        "pipe": pipe,
        "index": index,
    }

    chat = partial(_chat, statics)
    iface = gr.ChatInterface(
        fn=chat,
        title="Simple Chatbot",
        description="Enter your message and get a response from the AI.",
    )

    iface.launch(share=True)


if __name__ == "__main__":
    _main()

