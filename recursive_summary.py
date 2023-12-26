import math, os
from dotenv import load_dotenv
import os
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from huggingface_hub import InferenceClient

# from transformers import BartTokenizer
import tiktoken

MAX_INPUT_SIZE = 30000
MAX_OUTPUT_SIZE = 4096
MIN_OUTPUT_SIZE = 50


def get_client():
    print("Connecting to client ...")
    try:
        from dotenv import load_dotenv
        from pathlib import Path

        load_dotenv(Path(".env"))
        API_TOKEN = st.secrets["GOOGLE_API_KEY"]
    except:
        import streamlit as st

        API_TOKEN = st.secrets["GOOGLE_API_KEY"]

    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    return llm


class Summarizer(object):
    def __init__(self):
        self.client = get_client()
        self.tok = tiktoken.get_encoding("cl100k_base")
        self.tok_len_of = lambda x: len(self.tok.encode(x))

    def API_call(self, text):
        print("API call ->>>>>>>>>>>>>>> input length:", self.tok_len_of(text))
        summary = self.client.invoke(
            "This is a youtube video, provide concise summary of the video subtitles and highlight main topics discussed in the video: \n"
            + text
        ).content
        # print("API response <<<<<<<<<<<<<-", summary)
        return summary

    def adaptive_chunkify_bart(self, text):
        if self.tok_len_of(text) <= MAX_OUTPUT_SIZE:
            return [text]
        n_chunks = math.ceil(self.tok_len_of(text) / MAX_INPUT_SIZE)
        chunk_size = math.ceil(self.tok_len_of(text) / n_chunks) + 200
        print(f"{n_chunks=}, {chunk_size=}")

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=50,
            is_separator_regex=False,
        )
        chunks = list(
            map(lambda page: page.page_content, text_splitter.create_documents([text]))
        )
        print("Chunks sizes are:", [self.tok_len_of(c) for c in chunks])
        return chunks

    def summarize(self, comprehension):
        chunks = self.adaptive_chunkify_bart(comprehension)
        if len(chunks) == 1:
            return self.API_call(chunks[0])
        chunk_summaries = [self.API_call(chunk) for chunk in chunks]
        return self.summarize(" ".join(chunk_summaries))


if __name__ == "__main__":
    print(Summarizer().summarize(open("text.txt").read()))
