from langchain_core.prompts import ChatPromptTemplate

template = """
   - count_constraint: For "more than 3 people" use {{"gte": 3}}, for "exactly 2" use {{"eq": 2}}
"""

try:
    prompt = ChatPromptTemplate.from_template(template)
    print("Variables:", prompt.input_variables)
except Exception as e:
    print("Error:", e)
