import math
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient
import tiktoken

MAX_INPUT_SIZE = 30000
MAX_OUTPUT_SIZE = 4096
MIN_OUTPUT_SIZE = 50


def get_client():
    print("Connecting to client...")
    load_dotenv()  # Load environment variables once
    API_TOKEN = st.secrets["GOOGLE_API_KEY"]  # Prioritize Streamlit secrets
    if not API_TOKEN:
        API_TOKEN = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(model="gemini-pro")


class Summarizer:
    __instance = None

    @staticmethod
    def get_singleton():
        """Static access method."""
        if Summarizer.__instance == None:
            Summarizer()
        return Summarizer.__instance

    def __init__(self):
        """Virtually private constructor."""
        if Summarizer.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            self.client = get_client()
            self.tok = tiktoken.get_encoding("cl100k_base")
            self.tok_len_of = lambda x: len(self.tok.encode(x))
            Summarizer.__instance = self

    def API_call(self, text):
        print(f"API call ->>>>>>>>>>>>>>> input length: {self.tok_len_of(text)}")
        summary = self.client.invoke(
            "This is a youtube video, provide concise summary of the video subtitles and highlight main topics discussed in the video: \n"
            + text
        ).content
        return summary

    def adaptive_chunkify_bart(self, text):
        if self.tok_len_of(text) <= MAX_OUTPUT_SIZE:
            return [text]
        n_chunks = math.ceil(self.tok_len_of(text) / MAX_INPUT_SIZE)
        chunk_size = math.ceil(self.tok_len_of(text) / n_chunks) + 200
        print(f"{n_chunks=}, {chunk_size=}")

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=50, is_separator_regex=False
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
