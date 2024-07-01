import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

import warnings
import pprint
warnings.filterwarnings("ignore")


# making vector store

# Load documents
documents = SimpleDirectoryReader("/content/data").load_data()

# Create embedding model
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")

# Create vector store
d = 768
faiss_index = faiss.IndexFlatL2(d)
# dimension should match your embedding model
vector_store = FaissVectorStore(faiss_index=faiss_index)

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    vector_store=vector_store
)


def get_top_k_matches(query, k=3):
    """
    function to retrieve top k matches from vector store
    """
    retriever = index.as_retriever(similarity_top_k=k)
    nodes = retriever.retrieve(query)
    return [node.node.text for node in nodes]


# Loading model and tokenizer

checkpoint = "microsoft/phi-1_5"
model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# formatting prompt
SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer."""


def get_top_k_matches(query, k=3):
    retriever = index.as_retriever(similarity_top_k=k)
    nodes = retriever.retrieve(query)
    # return nodes
    return [node.node.text for node in nodes]


def format_prompt(prompt, retrieved_docs, k):
    PROMPT = f"Question:{prompt}\nContext:"
    for text in retrieved_docs:
        PROMPT += f"{text}\n"
    return PROMPT


# Generate function
def generate(formatted_prompt):
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


def main():
    query = input("Enter your query: ")
    retrieved_docs = get_top_k_matches(query)
    formatted_prompt = format_prompt(SYS_PROMPT, retrieved_docs, 3)
    response = generate(formatted_prompt)
    pprint.pprint(response)


if __name__ == '__main__':
    main()
