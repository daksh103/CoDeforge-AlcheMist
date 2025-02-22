import streamlit as st
import os
import json
import time
import random
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Load API keys from secrets
e2b_api_key = st.secrets["E2B_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]
google_api_key = st.secrets["GOOGLE_API_KEY"]

# Get Groq API keys and pick a random one
groq_api_keys = st.secrets["GROQ_API_KEY"].values()
selected_groq_api_key = random.choice(list(groq_api_keys))

# Set API keys as environment variables if needed
os.environ["GOOGLE_API_KEY"] = google_api_key

st.set_page_config(page_title="CoDeforge AlcheMist", page_icon="⚒️")
st.title("🛠️ CoDeforge AlcheMist")

# Define models
model_a = "gemma2-9b-it"  # Fixed model for Model A
models_for_model_m = ["llama3-70b-8192"]  # Model M selection

def get_model_a():
    return model_a

def get_model_m():
    return random.choice(models_for_model_m)

selected_model_m = get_model_m()
llm_a = ChatGroq(groq_api_key=selected_groq_api_key, model_name=get_model_a())
llm_m = ChatGroq(groq_api_key=selected_groq_api_key, model_name=selected_model_m)

prompt = ChatPromptTemplate.from_template(
    """
    You are an expert in transforming technical questions into clear, structured, and easily understandable language.

    **Format:**
    ```
    **Question Structure Overview**:
    q1
    q2
    q3

    **Updated Question 1**: [Rephrased version]
    ```
    Context:
    {context}
    **Question Number**: {input}
    """
)

model_m_prompt = ChatPromptTemplate.from_template(
    """
    You are an expert in generating complete, machine-readable C++ code for the provided questions.

    **Response Format:**
    ```
    **Question 1 - Code:**
    ```cpp
    [Full code solution]
    ```

    **Question 1 - Output:**
    ```
    [Expected output of the code]
    ```
    ```
    Context:
    {context}
    **Question Number**: {input}
    """
)

def extract_solutions(sections):
    solutions = []
    
    for section in sections:
        parts = section.split(" - Output:**", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            solutions.append(parts[1].strip())
    
    return solutions

def extract_questions_and_solutions(response_text):
    questions = []
    solutions = []

    sections = response_text.split("**Question ")
    print("Extracted Sections:", sections)  # Debugging statement

    extracted_solutions = extract_solutions(sections)  # Extract solutions separately

    for i, section in enumerate(sections[1:]):
        parts = section.split(" - Code:**")
        if len(parts) < 2:
            continue
        
        question_number = parts[0].strip()
        code_output_parts = parts[1].split(" - Output:**")

        code = code_output_parts[0].replace("```cpp", "").replace("```", "").strip()
        output = extracted_solutions[i] if i < len(extracted_solutions) else "Expected output not provided."

        questions.append(f"Question {question_number}")
        solutions.append((code, output))

    print("Extracted Questions:", questions)  # Debugging statement
    print("Extracted Solutions:hiiiiiiiiiiiiiiiiiiiiiiii", solutions)  # Debugging statement

    return questions, solutions

def generate_output_screenshot(output_text, question_number):
    """Creates an image of the expected output and saves it."""
    img_width, img_height = 400, 200
    #img = Image.new("RGB", (img_width, img_height), "dark blue")
    img = Image.new("RGB", (img_width, img_height), "#1C2130")

    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    if output_text=='Expected output not provided.':
        output_text="=== Code Execution Successful ==="
        x=output_text
        
    else:
        output_text=output_text[3:]
        output_text=output_text[:-3]
        x = output_text + "\n\n" + "=== Code Execution Successful ==="

    draw.text((20, 50), x, fill="white", font=font)  # First text
    #draw.text((20, 70), '=== Code Execution Successful ===', fill="grey", font=font)  # Second text moved down


    image_path = f"output_screenshot_q{question_number}.png"
    img.save(image_path)
    return image_path

def generate_pdf(questions, solutions):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, " Solutions", ln=True, align="C")
    pdf.ln(10)

    for i, question in enumerate(questions):
        pdf.set_font("Arial", style="B", size=12)
        pdf.multi_cell(0, 10, question)
        pdf.ln(2)

        pdf.set_font("Courier", size=10)
        pdf.multi_cell(0, 5, "Code:", border=1)
        pdf.multi_cell(0, 5, solutions[i][0])
        pdf.ln(2)

        output_text = solutions[i][1] if solutions[i][1].strip() else ""#
        output_image_path = generate_output_screenshot(output_text, i + 1)

        pdf.multi_cell(0, 5, "Output:", border=0)
        pdf.image(output_image_path, x=10, w=100)  
        

        pdf.ln(10)

    pdf_file_path = "solutions.pdf"
    pdf.output(pdf_file_path)
    return pdf_file_path

def process_uploaded_pdf(uploaded_file):
    """Processes uploaded PDF and generates structured questions and solutions."""
    if uploaded_file is None:
        st.error("⚠️ Please upload a PDF file")
        return

    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("temp_uploaded.pdf")
    docs = loader.load()

    if not docs:
        st.error("No content found in the uploaded PDF!")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)

    if not final_documents:
        st.error("No content found after text splitting!")
        return

    context = " ".join([doc.page_content for doc in final_documents])

    document_chain = create_stuff_documents_chain(llm_a, prompt)
    retriever = FAISS.from_documents(final_documents, GoogleGenerativeAIEmbeddings(model="models/embedding-001")).as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start_time = time.process_time()
    response = retrieval_chain.invoke({'input': "all", 'context': context})
    elapsed_time = time.process_time() - start_time
    response_text = response['answer']

    # Print response text to terminal
    print("\n======= Response Text =======\n")
    print(response_text)
    print("\n============================\n")

    code_chain = create_stuff_documents_chain(llm_m, model_m_prompt)
    code_retriever = FAISS.from_documents(final_documents, GoogleGenerativeAIEmbeddings(model="models/embedding-001")).as_retriever()
    code_retrieval_chain = create_retrieval_chain(code_retriever, code_chain)

    start_time_code = time.process_time()
    code_response = code_retrieval_chain.invoke({'input': "all", 'context': response_text})
    elapsed_time_code = time.process_time() - start_time_code
    code_solution = code_response['answer']

    # Print code solution to terminal
    print("\n======= Code Solution =======\n")
    print(code_solution)
    print("\n============================\n")

    questions, solutions = extract_questions_and_solutions(code_solution)

    pdf_file_path = generate_pdf(questions, solutions)
    with open(pdf_file_path, "rb") as pdf_file:
        st.download_button("📥 Download Solutions PDF", pdf_file, "solutions.pdf", "application/pdf")

    st.write("⏱ **Response Time for PDF Processing:**", elapsed_time, "seconds")
    st.write("⏱ **Response Time for Code Generation:**", elapsed_time_code, "seconds")
    st.write("🤖 **Model D (Question Processing):** ", get_model_a())
    st.write("🤖 **Model M (Code Generation):** ", selected_model_m)
    st.subheader("Code Solutions")
    st.markdown(code_solution)
 
uploaded_file = st.file_uploader("📂 Upload a PDF Document", type=["pdf"])

if uploaded_file is not None:
    process_uploaded_pdf(uploaded_file)



footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Made with ❤️ by Daksh Arora</p>
</div>
"""

st.markdown(footer,unsafe_allow_html=True)
