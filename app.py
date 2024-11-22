import os
import streamlit as st
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from document_processors import load_multimodal_data, load_data_from_directory
from utils import set_environment_variables
from io import BytesIO  # Importing for file download
from pylatex import Document, Section, Command, NewPage, Package, Figure
from pylatex.utils import NoEscape
import os
import matplotlib.pyplot as plt
import re
import json
import fitz
import shutil

# Set up the page configuration
st.set_page_config(layout="wide")

# Initialize settings
def initialize_settings():
    Settings.embed_model = NVIDIAEmbedding(model="nvidia/nv-embedqa-e5-v5", truncate="END")
    Settings.llm = NVIDIA(model="meta/llama-3.1-70b-instruct")
    Settings.text_splitter = SentenceSplitter(chunk_size=800)

# Create index from documents
def create_index(documents):
    vector_store = MilvusVectorStore(
        host="127.0.0.1",
        port=19530,
        dim=1024,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index =  VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist()
    return index


def extract_references_from_pdfs():
    """
    Extracts the "References" section from each PDF in the "uploaded_documents" folder
    and returns them as a single string with sequential numbering across PDFs.

    Returns:
        str: A single string containing all extracted references, separated by new lines.
    """
    folder_path = "uploaded_documents"  # Default folder path
    references_list = []
    reference_number = 1  # Start numbering from 1

    # Loop through each PDF in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            
            # Open the PDF
            with fitz.open(pdf_path) as pdf:
                # Initialize a variable to store the full text
                full_text = ""
                # Loop through each page and extract text
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    full_text += page.get_text()

                # Use regex to find the "References" section as a heading
                references_match = re.search(r"(?:^|\n)References\s*\n(.*?)(?:\n[A-Z]|$)", full_text, re.DOTALL | re.IGNORECASE)
                
                if references_match:
                    # Extract references content, split into lines, and enumerate them
                    references_content = references_match.group(1).strip()
                    references_lines = references_content.splitlines()
                    
                    for line in references_lines:
                        if line.strip():  # Skip empty lines
                            # Add line with sequential numbering
                            references_list.append(f"[{reference_number}] {line.strip()}")
                            reference_number += 1
                else:
                    references_list.append(f"No References section found in {filename}.")

    # Join all references with newlines
    all_references = "\n".join(references_list)
    return all_references

# Generate review paper from query results using LLM model
def generate_review_paper_from_query(query, query_engine, output_filename='review_paper.pdf'):
    # Initialize LaTeX document with two-column IEEE-style formatting
    doc = Document(documentclass='article', document_options='a4paper,twocolumn')
    doc.packages.append(Package('geometry'))
    doc.preamble.append(NoEscape(r'\geometry{top=19mm, bottom=43mm, left=14.32mm, right=14.32mm, columnsep=4.22mm}'))
    doc.packages.append(Package('times'))
    doc.preamble.append(NoEscape(r'\usepackage{titlesec}'))
    doc.preamble.append(NoEscape(r'\titleformat{\section}{\normalfont\fontsize{10}{12}\scshape\centering}{\thesection}{1em}{\MakeUpperCase}'))
    doc.preamble.append(NoEscape(r'\titleformat{\subsection}{\normalfont\fontsize{10}{12}\itshape}{\thesubsection}{1em}{}'))
    doc.preamble.append(NoEscape(r'\titleformat{\subsubsection}{\normalfont\fontsize{10}{12}\itshape}{\thesubsubsection}{1em}{\ignorespaces}'))
    doc.preamble.append(NoEscape(r'\renewcommand{\thesection}{\Roman{section}}'))
    doc.preamble.append(NoEscape(r'\renewcommand{\thesubsection}{\Alph{subsection}.}'))
    doc.preamble.append(NoEscape(r'\renewcommand{\thesubsubsection}{\arabic{subsubsection})}'))
    doc.preamble.append(NoEscape(r'\pagestyle{empty}'))
    doc.preamble.append(Command('title', 'Review Paper'))
    doc.preamble.append(Command('author', 'Generated by LLM'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))

    # Define sections for the document
    sections = {
        "abstract": "Summarize the key points related to the research topic, including objectives, methodology, and findings.",
        "keywords": "Provide a list of relevant keywords.",
        "introduction": "Provide an introduction outlining the background, significance, and objectives of the research.",
        "references": "Provide only references or citations directly related to the topic, without headers or additional notes.",
        "results": "Describe the research findings with clear, factual details.",
        "Images": (
            "Provide structured numerical data relevant to the research topic, "
            "include specific labels for each axis, and provide a caption for the plot. "
            "Data should follow the format: (x_value, y_value). "
            "Label format should be: 'X-Label: [Label text]' and 'Y-Label: [Label text]'. "
            "Caption format should be: 'Caption: [Caption text]'."
        ),
        "conclusion": "Summarize the key findings and discuss broader implications."
    }

    # Initialize JSON data structure for storing image data, labels, and caption
    json_data = {
        "data": [],
        "labels": {
            "x_label": "",
            "y_label": ""
        },
        "caption": ""
    }

    plot_data = []
    references_data = {}

    # Generate content for each section using the LLM model
    for section, prompt_template in sections.items():
        if section == "references":
            # Generate the literature review content
            reference_text = extract_references_from_pdfs()
            print(reference_text)
            prompt = f"{query}. {prompt_template}. {reference_text} Please write this section in a professional, academic tone, with citations in a numbered format without additional text."
            response = query_engine.query(prompt)
            
            # Accumulate the full content for the literature review
            full_review = ""
            for token in response.response_gen:
                full_review += token

            # Updated regex to match both [1] and 1. citation formats
            citation_pattern = re.compile(r'(\[\d+\]|\d+\.)\s+(.*?)(?=(\[\d+\]|\d+\.)|$)', re.MULTILINE | re.DOTALL)
            valid_references = citation_pattern.findall(full_review)

            # Write each valid citation to the JSON file with sequential numbering but without extra numbers
            for i, (_, reference, *_) in enumerate(valid_references, start=1):
                references_data[f"[{i}]"] = reference.strip()

            # Write the filtered references to a JSON file
            if references_data:
                with open('references.json', 'w') as json_file:
                    json.dump(references_data, json_file, indent=4)
            else:
                print("No valid references found to write to JSON.")
                
        else:
            # Generate content for other sections using the query engine
            prompt = f"{query}. {prompt_template} Please write this section in a professional, academic tone."
            response = query_engine.query(prompt)

            # Accumulate the full content for the section
            full_summary = ""
            for token in response.response_gen:
                full_summary += token
            
            # Extract image data, labels, and caption if in the "Images" section
            if section == "Images":
                data_lines = [line for line in full_summary.splitlines() if re.search(r'\d', line)]
                label_lines = [line for line in full_summary.splitlines() if "Label:" in line]
                caption_lines = [line for line in full_summary.splitlines() if "Caption:" in line]

                # Process and store data points
                for line in data_lines:
                    values = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                    if len(values) == 2:
                        plot_data.append([float(values[0]), float(values[1])])
                        json_data["data"].append([float(values[0]), float(values[1])])

                # Process and store labels if available
                if label_lines and len(label_lines) >= 2:
                    json_data["labels"]["x_label"] = label_lines[0].split(":")[1].strip()
                    json_data["labels"]["y_label"] = label_lines[1].split(":")[1].strip()

                # Process and store caption if available
                if caption_lines:
                    json_data["caption"] = caption_lines[0].split(":", 1)[1].strip()

                # Save JSON data for images to file
                with open('image_data.json', 'w') as json_file:
                    json.dump(json_data, json_file, indent=4)


            # Add content to the LaTeX document
            with doc.create(Section(section.capitalize(), numbering=False)):
                doc.append(NoEscape(full_summary))

    # Read and add literature review from the JSON file to the LaTeX document
    if os.path.exists('references.json'):
        with open('references.json', 'r') as json_file:
            references_data = json.load(json_file)
            
            formatted_review = ""
            for index, review in references_data.items():
                formatted_review += f"{index} {review}\n\n"  # Format as "[1] review1", "[2] review2", etc.

            # Add formatted literature review to LaTeX document
            with doc.create(Section("References", numbering=False)):
                doc.append(NoEscape(formatted_review))

                
    # Generate plot if data was provided
    if plot_data and len(plot_data) > 1:
        # Retrieve labels and caption from the JSON data or use defaults if not available
        x_label = json_data["labels"]["x_label"] or "X-Axis"
        y_label = json_data["labels"]["y_label"] or "Y-Axis"
        plot_caption = json_data["caption"] or "Plot based on generated data."

        # Convert plot_data to x and y coordinates
        x_data, y_data = zip(*plot_data)

        # Plot the data using Matplotlib
        plt.figure()
        plt.plot(x_data, y_data, marker='o', linestyle='-', color='b')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title('Generated Plot based on Query Data')
        
        # Save the plot image
        plot_image_filename = 'plot_image.png'
        plt.savefig(plot_image_filename)

        # Embed the plot image in the LaTeX document
        with doc.create(Figure(position='h!')) as plot_fig:
            plot_fig.add_image(plot_image_filename, width=NoEscape(r'0.9\linewidth'))
            plot_fig.add_caption(plot_caption)


    # Generate the LaTeX file
    tex_filename = 'review_paper'
    doc.generate_tex(tex_filename)

    # Compile LaTeX to PDF without prompting for input
    os.system(f"pdflatex -interaction=nonstopmode {tex_filename}")

    print(f"PDF successfully generated: {output_filename}")
    print("Literature review data saved to references.json")

    return output_filename



# Main function to run the Streamlit app
def main():
    set_environment_variables()
    initialize_settings()

    documents = None  # Initialize documents

    col1, col2 = st.columns([1, 2])

    # folder_path = "uploaded_images"
    # folder_path1= "uploaded_documents"

    # # Check if the folder exists
    # if os.path.exists(folder_path):
    #     shutil.rmtree(folder_path)
    #     print(f"Folder '{folder_path}' and all its contents have been deleted.")
    
    # if os.path.exists(folder_path1):
    #     shutil.rmtree(folder_path1)
    #     print(f"Folder '{folder_path1}' and all its contents have been deleted.")
    
    SAVE_FOLDER = "uploaded_documents"  # Define the folder where files will be saved

    # Ensure the folder exists
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    with col1:
        st.title("Multimodal RAG")
        
        input_method = st.radio("Choose input method:", ("Upload Files", "Enter Directory Path"))
        
        if input_method == "Upload Files":
            uploaded_files = st.file_uploader("Drag and drop files here", accept_multiple_files=True)
            
            if uploaded_files and st.button("Process Files"):
                with st.spinner("Processing files..."):
                    saved_file_paths = []
                    
                    for uploaded_file in uploaded_files:
                        if hasattr(uploaded_file, "name"):
                            # If it's a file-like object, save it to the designated folder
                            file_path = os.path.join(SAVE_FOLDER, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            saved_file_paths.append(file_path)
                        elif isinstance(uploaded_file, str):
                            # If it's already a string path, just append it
                            saved_file_paths.append(uploaded_file)
                    
                    # Load and process the saved file paths
                    documents = load_multimodal_data(uploaded_files)
                    st.session_state['index'] = create_index(documents)
                    st.session_state['history'] = []
                    st.session_state['documents'] = documents
                    st.success("Files processed and index created!")
        else:
            directory_path = st.text_input("Enter directory path:")
            if directory_path and st.button("Process Directory"):
                if os.path.isdir(directory_path):
                    with st.spinner("Processing directory..."):
                        documents = load_data_from_directory(directory_path)
                        st.session_state['index'] = create_index(documents)
                        st.session_state['history'] = []
                        st.session_state['documents'] = documents
                        st.success("Directory processed and index created!")
                else:
                    st.error("Invalid directory path. Please enter a valid path.")


    with col2:
        if 'index' in st.session_state:
            st.title("Chat")
            if 'history' not in st.session_state:
                st.session_state['history'] = []

            query_engine = st.session_state['index'].as_query_engine(similarity_top_k=20, streaming=True)

            user_input = st.chat_input("Enter your query:")

            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for message in st.session_state['history']:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state['history'].append({"role": "user", "content": user_input})
                
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    response = query_engine.query(user_input)
                    for token in response.response_gen:
                        full_response += token
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                st.session_state['history'].append({"role": "assistant", "content": full_response})

            if st.button("Clear Chat"):
                st.session_state['history'] = []
                st.rerun()

        # Input for review paper generation query
        st.title("Generate Review Paper")
        
        if 'index' in st.session_state:
            paper_query = st.text_input("Enter text prompt for generating a review paper:")
        
            if paper_query and st.button("Generate Review Paper"):
                with st.spinner("Generating review paper..."):
                    query_engine = st.session_state['index'].as_query_engine(similarity_top_k=20, streaming=True)
                    output_filename = generate_review_paper_from_query(paper_query, query_engine)
                    
                    # Read the generated PDF file into BytesIO for download
                    with open(output_filename, 'rb') as f:
                        pdf_data = f.read()
                        buffer = BytesIO(pdf_data)

                    # Provide a download button for the generated PDF
                    st.download_button(
                        label="Download Review Paper PDF",
                        data=buffer,
                        file_name="review_paper.pdf",
                        mime="application/pdf"
                    )
        else:
            st.warning("Please process documents to create an index before generating a review paper.")

if __name__ == "__main__":
    main()
