import torch
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

def load_data(dir_path):
    documents = SimpleDirectoryReader(dir_path).load_data()
    print("Done loading data")
    return documents


def load_model():
    print("Loading models")
    checkpoint = "microsoft/phi-2"
    system_prompt="""
    You are a Q&A assistant. Your goal is to answer questions as
    accurately as possible based on the instructions and context provided.
    """
    query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name=checkpoint,
        model_name=checkpoint,
        device_map="auto",
        # uncomment this if using CUDA to reduce memory usage
        model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
    )

    embed_model=LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
    
    return embed_model, llm

def postprocess(response):
    return response

def generate(query_engine ,query):
    response = query_engine.query(query)
    response = postprocess(response)
    return response


def main(query):
    documents = load_data()
    embed_model, llm = load_model()

    print("making index...")

    service_context=ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )
    index=VectorStoreIndex.from_documents(documents,service_context=service_context)
    query_engine=index.as_query_engine()
    generate(query)


if __name__ == "__main__":
    inp = input("Enter your query")
    main(inp)