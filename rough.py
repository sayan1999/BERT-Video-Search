import pandas as pd
from urllib.parse import urlparse
import streamlit as st
from streamlit_player import st_player
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
from recursive_summary import Summarizer


# Initialize resources with caching
@st.cache_resource
def init():
    return Summarizer(), SentenceTransformer("msmarco-distilbert-base-dot-prod-v3")


# Function to summarize text
def summarize(text, summarizer):
    return summarizer.summarize(text)


# Function to download and parse subtitles from a YouTube video URL
@st.cache_data
def parse_subtitles(url):
    video_id = urlparse(url).query[2::]
    try:
        subtitles = YouTubeTranscriptApi.get_transcript(video_id)
    except NoTranscriptFound:
        subtitles = (
            YouTubeTranscriptApi.list_transcripts(video_id).translate("en").fetch()
        )  # Handle translation if needed
    return pd.DataFrame(subtitles)


# Function to create embeddings for subtitles and enable semantic search
@st.cache_data
def get_relevant_line(subtitle_df, searchphrase, model):
    encoded_subtitles = model.encode(subtitle_df.text.tolist())
    index = faiss.IndexIDMap(
        faiss.IndexFlatIP(768)
    )  # Create FAISS index for efficient search
    index.add_with_ids(encoded_subtitles, np.arange(len(subtitle_df)))

    query_vector = model.encode([searchphrase])
    top_results = index.search(query_vector, 6)[1].tolist()[0]  # Retrieve top 6 results
    return subtitle_df.iloc[top_results]


# Main application logic
if __name__ == "__main__":
    summarizer, model = init()

    st.title("Bert-Video-Search-and-Jump")
    st.write("**Semantically search YouTube videos and jump to relevant sections**")

    vid_url = st.text_input("YouTube Video URL")

    if vid_url:
        with st.container():
            st_player(vid_url, playing=True)

        searchphrase = st.text_input("Search keywords")

        if searchphrase:
            subtitle_df = parse_subtitles(vid_url)
            search_results = get_relevant_line(subtitle_df, searchphrase, model)

            if not search_results.empty:
                st.write("Relevant sections:")
                for text, start_time in zip(
                    search_results["text"], search_results["start"]
                ):
                    st.button(
                        "Jump to section",
                        on_click=lambda: st_player(
                            vid_url + "&t={}s".format(round(float(start_time))),
                            playing=True,
                        ),
                    )
                    st.markdown(text)
            else:
                st.write("No relevant sections found.")

        summary = summarize(subtitle_df.text.str.cat(), summarizer)
        if summary:
            st.subheader("Video Summary:")
            st.markdown(summary)
