import gradio as gr
import argparse

from main import main
from functools import partial

SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer."""


def generate_response(filepath, message):
    res = main(SYS_PROMPT=SYS_PROMPT,
               filepath=filepath,
               query=message)
    return res


def _chat(filepath, message, history):
    response = generate_response(SYS_PROMPT, filepath, message)
#   temp = {"msg": message, "history": history,
#           "response": response}
    return response


def _main():
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Simple CLI to take a string input")
    parser.add_argument('-f', '--filepath', type=str,
                        default="../content/data", help="file dir for PDF/txt.. files")

    args = parser.parse_args()
    filepath = args.filepath

    chat = partial(_chat, filepath)
    iface = gr.ChatInterface(
        fn=chat,
        title="Simple Chatbot",
        description="Enter your message and get a response from the AI.",
    )

    iface.launch(share=True)


if __name__ == "__main__":
    _main()
