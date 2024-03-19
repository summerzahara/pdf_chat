import streamlit as st
from llm_helper import read_pdf, split_text, embed_chunks, retrieve_data
from icecream import ic


def create_db(pdf):
    text=read_pdf(pdf)
    chunks=split_text(text)
    vector_store=embed_chunks(chunks)
    return vector_store


def main():
    st.title("PDF Chat")
    my_pdf = st.file_uploader(
        "Upload File:",
        type=["pdf"],
        accept_multiple_files=False,
    )
    submit = st.button(
        "Submit",
        type="primary"
    )
    with st.form("query form"):
        user_query = st.text_input("Enter Query")
        submit_query = st.form_submit_button("Query")

    if submit:
        db = create_db(my_pdf)
        st.write("DB created")

    if submit_query:
        ic("test-retrieve")
        result = retrieve_data(db,user_query)
        ic(result)
        


if __name__ == "__main__":
    main()

