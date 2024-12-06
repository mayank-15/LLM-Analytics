import pandas as pd
import spacy

# def process_entities(csv_file_path):
#     df = pd.read_csv(csv_file_path)
#     # print(df)

#     # Use spaCy NER model to extract entities from columns
#     nlp_ner = spacy.load("model-best")
#     entities_to_remove = []
#     for column in df.columns:
#         doc = nlp_ner(column)
#         for ent in doc.ents:
#             entities_to_remove.append(ent.text)
#     # print(f"Entities to remove: {entities_to_remove}")
    
#     # Remove columns with entities from the dataframe
#     df = df.drop(columns=entities_to_remove, errors='ignore')
    
#     # Save the modified dataframe back to a CSV
#     entities_csv_file_path = "modified.csv"
#     df.to_csv(entities_csv_file_path, index=False)

#     return entities_csv_file_path

def process_entities(csv_file_path):
    return csv_file_path

def generate_template_yaml(modified_csv_file_path):
    df = pd.read_csv(modified_csv_file_path)
    num_rows = len(df)
    num_columns = len(df.columns)

    # Start constructing the YAML-like string
    template = "df:\n"
    template += "  name: null\n"
    template += "  description: null\n"
    template += f"  type: pd.DataFrame\n"
    template += f"  rows: {num_rows}\n"
    template += f"  columns: {num_columns}\n"
    template += "  schema:\n"
    template += "    fields:\n"

    for column in df.columns:
        dtype = df[column].dtype
        samples = df[column].sample(n=3).tolist()

        template += f"    - name: {column}\n"
        template += f"      type: {str(dtype)}\n"
        template += "      samples:\n"
        for sample in samples:
            template += f"      - {sample}\n"

    return template

# import pandas as pd

# def generate_template_yaml(csv_file_path):
#     df = pd.read_csv(csv_file_path)
#     num_rows = len(df)
#     num_columns = len(df.columns)

#     # Start constructing the YAML-like string
#     template = "df:\n"
#     template += "  name: null\n"
#     template += "  description: null\n"
#     template += f"  type: pd.DataFrame\n"
#     template += f"  rows: {num_rows}\n"
#     template += f"  columns: {num_columns}\n"
#     template += "  schema:\n"
#     template += "    fields:\n"

#     for column in df.columns:
#         dtype = df[column].dtype
#         samples = df[column].sample(n=3).tolist()

#         template += f"    - name: {column}\n"
#         template += f"      type: {str(dtype)}\n"
#         template += "      samples:\n"
#         for sample in samples:
#             template += f"      - {sample}\n"

#     return template

def edit_generated_prompt(template):
    text = """
    Update this initial code:python
    
    import pandas as pd
    
    # Write code here
    
    # Declare result var: 
    type (possible values "string", "number", "dataframe", "plot"). Examples: {{ "type": "string", "value": f"The highest salary is {{highest_salary}}." }} or {{ "type": "number", "value": 125 }} or {{ "type": "dataframe", "value": pd.DataFrame({{...}}) }} or {{ "type": "plot", "value": "temp_chart.png" }}
    
    
    ### QUERY
    {user_input}
    
    Variable df: list[pd.DataFrame] is already declared.
    
    At the end, declare "result" variable as a dictionary of type and value. Print the result.value
    
    If you are asked to plot a chart, use "matplotlib" for charts, save as png.
    
    
    Generate python code and return full updated code:
    """
    template += text
    # print("[generate_template]", template)
    return template

# modified_csv_path = process_entities(csv_file_path)
# template = generate_template_yaml(modified_csv_path)