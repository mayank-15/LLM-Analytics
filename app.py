import streamlit as st
import os
import tempfile
import gc
import base64
import time
import pandas as pd
from PIL import Image
from dotenv import load_dotenv

# ----- CSV-Related Imports -----
from openai import OpenAI
from generate_template import process_entities, generate_template_yaml, edit_generated_prompt
from utils import extract_code, initialize_state
from predefined_prompts import prompts as predefined_prompts

# ----- PDF-Related Imports -----
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
from agentic_rag.tools.custom_tool import DocumentSearchTool

# Load environment variables (for CSV processing)
load_dotenv(".env")

# ============================
# Sidebar: Mode Selection
# ============================
st.sidebar.header("Select Processing Mode")
mode = st.sidebar.radio("Choose a mode:", options=["CSV", "PDF"])

# ============================
# CSV Processing Mode
# ============================
if mode == "CSV":
    st.markdown("# Abhiyan GPT")
    
    # Initialize any necessary state for CSV mode
    initialize_state()
    if "openai" not in st.session_state:
        st.session_state["openai"] = "o1-mini"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload CSV", key="csv")
    
    if uploaded_file:
        # Read CSV and save locally
        df = pd.read_csv(uploaded_file)
        csv_path = "uploaded.csv"
        df.to_csv(csv_path, index=False)
        
        # Process the CSV file and generate the prompt template
        modified_csv_path = process_entities(csv_path)
        template = generate_template_yaml(modified_csv_path)
        prompt_template = edit_generated_prompt(template)
        
        # Display existing conversation (if any)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # --- Predefined Prompt Buttons ---
        cols = st.columns(2)
        button_cols = iter(cols)
        for prompt_name, prompt_text in predefined_prompts.items():
            col = next(button_cols, None)
            if col is None:
                cols = st.columns(2)
                button_cols = iter(cols)
                col = next(button_cols)
            
            if col.button(prompt_name):
                with st.spinner(f"Processing '{prompt_name}'..."):
                    st.session_state.last_button_clicked = prompt_name
                    st.session_state.messages.append({"role": "user", "content": prompt_text})
                    
                    final_prompt = prompt_template.format(user_input=prompt_text)
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    completion = client.chat.completions.create(
                        model=st.session_state["openai"],
                        messages=[{"role": "user", "content": final_prompt}]
                    )
                    
                    responses = extract_code(completion, modified_csv_path)
                    for response in responses:
                        if response["type"] == "plot":
                            st.image(response["value"])
                        else:
                            st.write(response["value"])
                        st.session_state.messages.append({"role": "assistant", "content": response["value"]})
                    
                    st.session_state.last_button_clicked = None
        
        # --- Chat Input for User-Generated Prompts ---
        user_prompt = st.chat_input("Enter Prompt")
        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)
            
            final_prompt = prompt_template.format(user_input=user_prompt)
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            completion = client.chat.completions.create(
                model=st.session_state["openai"],
                messages=[{"role": "user", "content": final_prompt}]
            )
            responses = extract_code(completion, modified_csv_path)
            for response in responses:
                if response["type"] == "plot":
                    st.image(response["value"])
                else:
                    st.write(response["value"])
                st.session_state.messages.append({"role": "assistant", "content": response["value"]})
    else:
        st.info("Please upload a CSV file to start processing.")

# ============================
# PDF Processing Mode
# ============================
elif mode == "PDF":
    st.markdown("# Abhiyan GPT")
    
    # Initialize PDF-specific state if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Chat history
    if "pdf_tool" not in st.session_state:
        st.session_state.pdf_tool = None  # To store the DocumentSearchTool
    if "crew" not in st.session_state:
        st.session_state.crew = None      # To store the Crew object

    def reset_chat():
        st.session_state.messages = []
        gc.collect()

    def display_pdf(file_bytes: bytes, file_name: str):
        """Displays the uploaded PDF in an iframe."""
        base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
        pdf_display = f"""
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" 
            height="600px" 
            type="application/pdf"
        ></iframe>
        """
        st.markdown(f"### Preview of {file_name}")
        st.markdown(pdf_display, unsafe_allow_html=True)

    def create_agents_and_tasks(pdf_tool):
        """Creates a Crew with the given PDF tool and a web search tool."""
        web_search_tool = SerperDevTool()
        
        retriever_agent = Agent(
            role="Retrieve relevant information to answer the user query: {query}",
            goal=(
                "Retrieve the most relevant information from the available sources "
                "for the user query: {query}. Always try to use the PDF search tool first. "
                "If you are not able to retrieve the information from the PDF search tool, "
                "then try to use the web search tool."
            ),
            backstory=(
                "You're a meticulous analyst with a keen eye for detail. "
                "You're known for your ability to understand user queries: {query} "
                "and retrieve knowledge from the most suitable knowledge base."
            ),
            verbose=True,
            tools=[t for t in [pdf_tool, web_search_tool] if t],
        )
        
        response_synthesizer_agent = Agent(
            role="Response synthesizer agent for the user query: {query}",
            goal=(
                "Synthesize the retrieved information into a concise and coherent response "
                "based on the user query: {query}. If you are not able to retrieve the "
                "information then respond with \"I'm sorry, I couldn't find the information "
                "you're looking for.\""
            ),
            backstory=(
                "You're a skilled communicator with a knack for turning "
                "complex information into clear and concise responses."
            ),
            verbose=True
        )
        
        retrieval_task = Task(
            description=(
                "Retrieve the most relevant information from the available "
                "sources for the user query: {query}"
            ),
            expected_output=(
                "The most relevant information in the form of text as retrieved "
                "from the sources."
            ),
            agent=retriever_agent
        )
        
        response_task = Task(
            description="Synthesize the final response for the user query: {query}",
            expected_output=(
                "A concise and coherent response based on the retrieved information "
                "from the right source for the user query: {query}. If you are not "
                "able to retrieve the information, then respond with: "
                "\"I'm sorry, I couldn't find the information you're looking for.\""
            ),
            agent=response_synthesizer_agent
        )
        
        crew = Crew(
            agents=[retriever_agent, response_synthesizer_agent],
            tasks=[retrieval_task, response_task],
            process=Process.sequential,
            verbose=True
        )
        return crew

    # --- Sidebar: PDF Upload and Clear Chat ---
    st.sidebar.header("Add Your PDF Document")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"], key="pdf")
    
    if uploaded_file is not None:
        if st.session_state.pdf_tool is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                with st.spinner("Indexing PDF... Please wait..."):
                    st.session_state.pdf_tool = DocumentSearchTool(file_path=temp_file_path)
            st.success("PDF indexed! Ready to chat.")
        display_pdf(uploaded_file.getvalue(), uploaded_file.name)
    
    st.sidebar.button("Clear Chat", on_click=reset_chat)

    # --- Main Chat Interface for PDF ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    prompt = st.chat_input("Ask a question about your PDF...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if st.session_state.crew is None:
            st.session_state.crew = create_agents_and_tasks(st.session_state.pdf_tool)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Thinking..."):
                inputs = {"query": prompt}
                result = st.session_state.crew.kickoff(inputs=inputs).raw
            
            # Simulate a typing effect
            lines = result.split('\n')
            for i, line in enumerate(lines):
                full_response += line
                if i < len(lines) - 1:
                    full_response += '\n'
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.15)
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": result})
