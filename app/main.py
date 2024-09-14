from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Request,
    WebSocket,
    status,
    WebSocketDisconnect,
    BackgroundTasks,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from langchain_cohere import CohereEmbeddings
from langchain_milvus import Milvus
from utils import load_split_file, call_openai
from dotenv import load_dotenv
import os

import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()
templates = Jinja2Templates(directory="../templates")
app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY"),
)
with open("prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()


def create_db_from_file(uploaded_file):
    docs = load_split_file(f"{uploaded_file.filename}")
    vector_store_saved = Milvus.from_documents(
        docs,
        embeddings,
        collection_name="langchain_example",
        connection_args={
            "uri": os.getenv("MILVUS_URI"),
            "token": os.getenv("MILVUS_TOKEN"),
            "secure": True,
        },
    )
    print("Vector store saved")


@app.get("/upload", response_class=HTMLResponse)
def return_homepage(request: Request):
    return templates.TemplateResponse(request=request, name="upload.html")


@app.post("/upload")
def upload_pdf_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if file.filename.endswith((".pdf", ".docx")):
        contents = file.file.read()
        with open(f"{file.filename}", "wb") as f:
            f.write(contents)
        file.file.close()
    background_tasks.add_task(create_db_from_file, file)
    return RedirectResponse(
        url="/",
        status_code=status.HTTP_303_SEE_OTHER,
        background=background_tasks,
    )


@app.get("/", response_class=HTMLResponse)
def return_homepage(request: Request):
    return templates.TemplateResponse(request=request, name="chatting.html")


vector_store_loaded = Milvus(
    embeddings,
    connection_args={
        "uri": os.getenv("MILVUS_URI"),
        "token": os.getenv("MILVUS_TOKEN"),
        "secure": True,
    },
    collection_name="langchain_example",
)


@app.websocket("/")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    user_messages = []
    try:
        while True:
            user_input = await websocket.receive_text()
            results = vector_store_loaded.similarity_search(
                user_input,
                k=5,
            )
            page_content = "\n".join([i.page_content for i in results])
            user_messages.append(user_input)
            user_messages_str = "\n".join(user_messages)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"### **User Query:** \n{user_messages_str} \n### **Retrieved Context:** \n{page_content}",
                },
            ]
            response = call_openai(client, messages)
            await websocket.send_text(response)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
