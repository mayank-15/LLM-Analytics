import re
import sys
import io
import streamlit as st


def display_chat_history():
    """
    Display chat history
    :return:
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def initialize_state():
    if 'last_button_clicked' not in st.session_state:
        st.session_state['last_button_clicked'] = None


def extract_code(completion, modified_csv_path):
    """
    Extract Python code from the OpenAI response and execute the code.
    :param completion: OpenAI completion Object
    :param modified_csv_path: Upload CSV file path

    :return: Output from the extracted Python code.
    """
    response = completion.choices[0].message.content
    captured_output = []

    code_blocks = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
    for block in code_blocks:
        # Ensure 'import pandas as pd' is at the beginning
        import_statement = 'import pandas as pd\n'
        if 'import pandas as pd' not in block:
            block = import_statement + block
        else:
            # Remove any existing import statements to avoid duplicates
            block = block.replace('import pandas as pd\n', '')
            block = import_statement + block

        # Ensure the DataFrame read line is correctly placed after the import statement
        new_line = f"df = pd.read_csv('{modified_csv_path}')\n"
        if "# Write code here" in block:
            marker_index = block.find("# Write code here") + len("# Write code here")
            modified_code = block[:marker_index] + '\n' + new_line + block[marker_index:]
        else:
            modified_code = import_statement + new_line + block

        # Execute the modified code block
        try:
            local_vars = {}
            exec(modified_code, globals(), local_vars)
            captured_output.append(local_vars.get('result', 'No result variable defined.'))
        except Exception as e:
            captured_output.append(f"Error executing code: {e}")
        # print('utils', captured_output)    

    return captured_output
