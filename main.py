import streamlit as st
import os
from utils import *

def main():
    st.set_page_config(page_title="PDF summarizer")

    st.title("PDF Summarizing Tool")
    st.write("Summarize your pdf files in just seconds")
    st.divider() #space

    pdf = st.file_uploader("Upload you PDF doc", type='pdf')

    submit = st.button("Generate summary")

    os.environ["OPENAI_API_KEY"] = "********"

    if submit:
        response = summarizer(pdf)

        st.subheader("Summary of file:")
        st.write(response)

if __name__ == "__main__":
    main()