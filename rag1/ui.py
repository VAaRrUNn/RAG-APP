import gradio as gr
import os
import shutil
import argparse
from functools import partial

from .model import test_model, gen, load_model
from .io import preprocessing

flag = 0

statics = {
        "pipe": None,
        "index": None,
        "flag": 0,
        "filepath": None,
    }

def generate_response(statics, message):
    res = gen(pipe = statics["pipe"],
              index = statics["index"],
              query=message)
    return res

def chat(message, history):
    global static
    if statics["flag"] == 0:
        index, vector_store = preprocessing(
                    filepath=statics["filepath"])
        
        if index is not None:
            statics["flag"] = 1

        statics["index"] = index

    response = generate_response(statics, message)
    return response

def save_files(files):
    if files is not None and len(files) > 0:
        saved_files = []
        for file in files:
            # Get the original filename
            original_name = os.path.basename(file.name)
            # Create the destination path
            dest_path = os.path.join('data', original_name)
            # Copy the file to the data directory
            shutil.copy(file.name, dest_path)
            saved_files.append(original_name)
        return f"Files uploaded successfully to the 'data' folder: {', '.join(saved_files)}"
    return "No files were uploaded."


def ui(chat):
    with gr.Blocks() as demo:
        gr.Markdown("# RAG chatbot")
        
        with gr.Row():
            file_output = gr.Textbox(label="File Upload Status")
            upload_button = gr.File(label="Click to Upload", file_types=[".pdf", ".txt", ".doc", ".docx"], file_count="multiple")
        
        upload_button.change(save_files, upload_button, file_output)
        
        chatbot = gr.ChatInterface(
            fn=chat,
            title="Chat Interface",
            description="Enter your message and get a response from the AI.",
        )

    demo.launch()

def _main():

    global statics

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Simple CLI to take a string input")
    parser.add_argument('-f', '--filepath', type=str,
                        default="data", help="file dir for PDF/txt.. files")

    args = parser.parse_args()
    statics["filepath"] = args.filepath
    
    # Ensure the filepath exists
    if not os.path.exists(statics["filepath"]):
        os.makedirs(statics["filepath"])

    # load model
    statics["pipe"] = load_model()
    ui(chat)


if __name__ == "__main__":
    _main()








