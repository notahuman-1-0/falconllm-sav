

import streamlit as st
from langchain import HuggingFaceHub
from apikey import apikey
from langchain import PromptTemplate, LLMChain
import os

# Set Hugging Face Hub API token
# Make sure to store your API token in the `apikey_hungingface.py` file
os.environ["HUGGINGFACEHUB_API_TOKEN"] = apikey 

# Set up the language model using the Hugging Face Hub repository
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.3, "max_new_tokens": 2000})

# Set up the prompt template
template = """
I'm an artificial intelligence assistant.
I give you helpful, detailed, and polite answers to your questions
Question: {question}\n\nAnswer: Hmmmm, ok let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Create the Streamlit app
def main():
    
    st.title("savbot - your helpful assistant <3 by srini")

    # Get user input
    question = st.text_input("what's up?")

    # Generate the response
    if st.button("help me out!"):
        with st.spinner("here you go......"):
            response = llm_chain.run(question)
        st.success(response)

if __name__ == "__main__":
    main()