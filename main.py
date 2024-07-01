import argparse
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

import warnings
import pprint
warnings.filterwarnings("ignore")


def load_models():
    checkpoint = "microsoft/phi-1_5"
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Models are runnning on {device}")
    return (model, tokenizer)


def make_index(filepath):
    print(f"loading data")
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


def format_prompt(prompt, retrieved_docs, k):
    PROMPT = f"Question:{prompt}\nContext:"
    for text in retrieved_docs:
        PROMPT += f"{text}\n"
    return PROMPT


# Generate function
def generate(model, tokenizer, SYS_PROMPT, formatted_prompt):
    formatted_prompt = formatted_prompt[:2000]  # to avoid GPU OOM
    messages = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": formatted_prompt}]
    # tell the model to generate
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


flag = 0


def preprocessing(filepath):
    index, model, tokenizer = None

    # Preprocessing and loading
    index, vector_store = make_index(filepath=filepath)
    model, tokenizer = load_models()

    return (model, tokenizer, index, vector_store)


def gen(model,
        tokenizer,
        index,
        query):
    # get all related docs
    retrieved_docs = get_top_k_matches(query, index)

    # generate output
    formatted_prompt = format_prompt(SYS_PROMPT, retrieved_docs, 3)
    response = generate(model, tokenizer, SYS_PROMPT, formatted_prompt)
    return response


SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer."""

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(
#         description="Simple CLI to take a string input")
#     parser.add_argument('-q', '--query', type=str,
#                         default="Tell me about y", help="Input query string")

#     parser.add_argument('-f', '--filepath', type=str,
#                         default="/content/data", help="file dir for PDF/txt.. files")

#     args = parser.parse_args()
#     query = args.query
#     filepath = args.filepath
