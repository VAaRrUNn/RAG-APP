import gradio as gr
import argparse

from main import preprocessing, gen, test_model
from functools import partial

SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer."""


def generate_response(statics, message):
    res = gen(**statics,
              query=message)
    return res


def _chat(statics, message, history):
    response = generate_response(statics, message)
#   temp = {"msg": message, "history": history,
#           "response": response}
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

    model = tokenizer = index = vector_store = None
    global flag
    if flag == 0:
        model, tokenizer, index, vector_store = preprocessing(
            filepath=filepath)
        test_model(model, tokenizer)
        flag = 1

    statics = {
        "model": model,
        "tokenizer": tokenizer,
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
