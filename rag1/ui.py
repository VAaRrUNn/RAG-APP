import gradio as gr
import os
import shutil
import argparse
from functools import partial

from main import preprocessing, test_model, gen

def generate_response(statics, message):
    res = gen(**statics,
              query=message)
    return res


def _chat(statics, message, history):
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

flag = 0

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
        # test_model(pipe)
        flag = 1

    statics = {
        "pipe": pipe,
        "index": index,
    }

    chat = partial(_chat, statics)
    
    # Ensure the 'data' directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    ui(chat)


if __name__ == "__main__":
    _main()







