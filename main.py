import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

import sys
import warnings
warnings.filterwarnings("ignore")


def load_models(device = None, checkpoint = None):
    # Config for quantization
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
        # llm_int8_enable_fp32_cpu_offload=True,
    )

    # Applying quantization if running on GPU
    checkpoint = "Qwen/Qwen2-7B-Instruct"
    checkpoint = "microsoft/phi-2"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    model = None
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                                      quantization_config=bnb_config,                                              
                                                      trust_remote_code=True,)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                                     trust_remote_code=True)
    
    return (model, tokenizer)

def test_model(model, tokenizer):
    """simple text function to test the working of model"""
    try:
        ...
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


def format_prompt(prompt, retrieved_docs, k):
    PROMPT = f"Question:{prompt}\nContext:"
    for text in retrieved_docs:
        PROMPT += f"{text}."
    return PROMPT


# Generate function
def generate(model, tokenizer, SYS_PROMPT, formatted_prompt):
    device = "cuda" if torch.cuda().is_available() else "cpu"
    # device = "cuda" # quantization only works on cuda for now
    formatted_prompt = formatted_prompt[:2000]  # to avoid GPU OOM
    messages = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": formatted_prompt}]
    # tell the model to generate
    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=1024
    )
    generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def preprocessing(filepath):
    index = model = tokenizer = None

    # Preprocessing and loading
    index, vector_store = setup_vector_store_index(filepath=filepath)
    model, tokenizer = load_models()

    return (model, tokenizer, index, vector_store)


def gen(model,
        tokenizer,
        index,
        query):
    # get all related docs
    retrieved_docs = get_top_k_matches(query, index) # list of string

    # generate output
    formatted_prompt = format_prompt(query, retrieved_docs, 3)
    response = generate(model, tokenizer, SYS_PROMPT, formatted_prompt)
    return response


SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer."""

