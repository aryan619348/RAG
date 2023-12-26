import gradio as gr
from ingest import configure_retriever
from chain import my_chain

def chatbot(input_text, history,uploaded_file):
    
    if uploaded_file is not None:
        ret=configure_retriever(uploaded_files=uploaded_file)
    else:
        ret =configure_retriever("")
    
    response =my_chain(ret,input_text)

    return response

demo = gr.ChatInterface(chatbot, 
                        additional_inputs=[
                            gr.File(file_types=["pdf", "csv"], file_count="multiple")
                        ],
                        title="RAG chain built using Langchain",
                        description="Upload your documents in the additional input section and enjoy",
                       )

demo.launch()

# import gradio as gr
# from ingest import configure_retriever
# from chain import my_chain

# def chatbot(input_text, uploaded_file):
#     # Your chatbot logic here
    
#     print("checkpoint1")
#     if uploaded_file is not None:
#         # Process the uploaded file (you can replace this with your own logic)
#         ret=configure_retriever(uploaded_files=uploaded_file)
    
#     response =my_chain(ret,input_text)

#     return response

# iface = gr.Interface(
#     fn=chatbot,
#     inputs=[
#         gr.Textbox(placeholder="Enter your text here"),
#         gr.UploadButton("Click to Upload a File", file_types=["pdf", "csv"], file_count="multiple")
#     ],
#     outputs=gr.Textbox(label="Response")
# )

# iface.launch()
