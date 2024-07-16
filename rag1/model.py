from transformers import BitsAndBytesConfig, pipeline
import torch 

import sys
import warnings
from .io import get_top_k_matches

warnings.filterwarnings("ignore")

SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question.
If you don't know the answer, just say "I do not know." Don't make up an answer.
Answer should be concise and write answer in conversational answer.
"""
def load_model(checkpoint = None,
              device = None):
    
    torch.cuda.empty_cache()

    if checkpoint is None:
        checkpoint = "microsoft/Phi-3-mini-4k-instruct"

    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
    
    # check device is not given explicitly
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"Loading model on: {device}")
    
    pipe = None
    
    if device == "cuda":
        print("In cuda")
        pipe = pipeline("text-generation", 
                        model=checkpoint,
                        trust_remote_code=True,
                        quantization_config=bnb_config,
                        model_kwargs={"device_map": device})
    else:
        print("In cpu")
        pipe = pipeline("text-generation", 
                        model=checkpoint,
                        trust_remote_code=True,
                        device=device,)
        print(f"Model is loaded on CPU, it may crash...")
        
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


def format_prompt(prompt, retrieved_docs):
    PROMPT = f"Question:{prompt}\nContext:"
    for text in retrieved_docs:
        PROMPT += f"{text}."
    return PROMPT


def generate(pipe, formatted_prompt, max_new_tokens=120, temperature=0.7, top_p=0.9, do_sample=True, repetition_penalty=1.1):
    formatted_prompt = formatted_prompt[:2000]  # to avoid OOM
    messages = f"System: {SYS_PROMPT}\n{formatted_prompt}\nAssistant:"
    
    model = pipe.model
    tokenizer = pipe.tokenizer
    
    input_ids = tokenizer.encode(messages, return_tensors="pt").to(model.device)
    full_response = ""
    
    while True:
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
#             pad_token_id=tokenizer.eos_token_id
        )
        
        token_ids = outputs.cpu().squeeze().tolist()
        end_token_id = pipe.tokenizer.eos_token_id
        n_token_ids = [n for n in token_ids if n!=end_token_id]
        
        # exclude input from the final result.
        full_sequence = pipe.tokenizer.decode(n_token_ids)
        new_part = full_sequence[len(pipe.tokenizer.decode(input_ids[0])):]
        full_response += new_part
        
        
        if (len(token_ids) != len(n_token_ids)):
            print("we are done")
            break

        input_ids = outputs
        
    
    return full_response


def gen(pipe,
        index,
        query):
    
    # No documents are provided by user
    if index is None:
        default_context = "There is no context provided, Answer based on your knowledge"
        formatted_prompt = format_prompt(query, [default_context])
        response = generate(pipe, formatted_prompt)
        return response
    
    # get all related docs
    retrieved_docs = get_top_k_matches(query, index) # list of string

    # generate output
    formatted_prompt = format_prompt(query, retrieved_docs)

    response = generate(pipe, formatted_prompt)
    return response