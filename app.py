import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
from generate_template import process_entities, generate_template_yaml, edit_generated_prompt
from utils import extract_code, initialize_state
from predefined_prompts import prompts as predefined_prompts
from PIL import Image
import pandas as pd 

# API
load_dotenv(".env")
API_KEY = os.getenv("OPENAI_API_KEY")

# st.image("assets/logo.png", width=400, use_column_width='auto')
client = OpenAI(api_key=API_KEY)

# Initialize state
initialize_state()

#uploaded_file = st.file_uploader(label="Upload CSV")
# uploaded_file_1 = st.file_uploader("Upload CSV 1", key="csv1")
# uploaded_file_2 = st.file_uploader("Upload CSV 2", key="csv2")
# if uploaded_file_1 and uploaded_file_2:
#     df_1 = pd.read_csv(uploaded_file_1)
#     df_2 = pd.read_csv(uploaded_file_2)
#     merged_df = pd.merge(df_1, df_2, left_on='order_id', right_on='gateway_order_id', how='inner')
#     merged_csv_path = "merged.csv"
#     merged_df.to_csv(merged_csv_path, index=False)

#     modified_csv_path = process_entities(csv_path)
#     template = generate_template_yaml(modified_csv_path)
#     prompt_template = edit_generated_prompt(template)

uploaded_file = st.file_uploader("Upload CSV", key="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    csv_path = "uploaded.csv"
    df.to_csv(csv_path, index=False)

    modified_csv_path = process_entities(csv_path)
    template = generate_template_yaml(modified_csv_path)
    prompt_template = edit_generated_prompt(template)

if "openai" not in st.session_state:
    st.session_state["openai"] = "gpt-4o"

# Initialize Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Predefined Prompts
cols = st.columns(2)
button_cols = iter(cols)

for prompt_name, prompt_text in predefined_prompts.items():
    col = next(button_cols, None)
    if col is None:
        cols = st.columns(2)
        button_cols = iter(cols)
        col = next(button_cols)

    if col.button(prompt_name):
        # Provide immediate feedback
        with st.spinner(f"Processing '{prompt_name}'..."):
            st.session_state.last_button_clicked = prompt_name
            st.session_state.messages.append({"role": "user", "content": prompt_text})

            final_prompt = prompt_template.format(user_input=prompt_text)
            completion = client.chat.completions.create(
                model=st.session_state["openai"],
                messages=[{"role": "user", "content": final_prompt}]
            )

            responses = extract_code(completion, modified_csv_path)
            # responses = extract_code(completion, uploaded_file.name)
            for response in responses:
                if response["type"] == "plot":
                    st.image(response["value"])
                else:
                    st.write(response["value"])
                    st.session_state.messages.append({"role": "assistant", "content": response["value"]})

            st.session_state.last_button_clicked = None

# User-generated prompt handling
if user_prompt := st.chat_input("Enter Prompt"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    final_prompt = prompt_template.format(user_input=user_prompt)
    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                "role": "user", "content": final_prompt
            }
        ]
    )

    responses = extract_code(completion, modified_csv_path)
    # responses = extract_code(completion, uploaded_file.name)
    for response in responses:
        if response["type"] == "plot":
            st.image(response["value"])
        else:
            st.write(response["value"])
        st.session_state.messages.append({"role": "assistant", "content": response["value"]})
