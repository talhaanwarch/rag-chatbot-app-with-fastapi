from typing import Annotated
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    Docx2txtLoader,
)


def load_split_file(file_path: Annotated[any, "file format should be .pdf"]):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx") or file_path.endswith(".doc"):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(
            "Unsupported file format. Only .pdf, .docx and .doc are supported."
        )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    data = loader.load_and_split(text_splitter)
    print("length of chunks", len(data))

    return data


def call_openai(client, messages):
    response = client.chat.completions.create(
        model="llama3-groq-70b-8192-tool-use-preview",
        temperature=0.5,
        messages=messages,
    )
    return response.choices[0].message.content
