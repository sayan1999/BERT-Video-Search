import random, traceback
from urllib.parse import urlparse
import pandas as pd
from streamlit_player import st_player
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
import streamlit_analytics
from recursive_summary import Summarizer
from youtube_transcript_api._errors import NoTranscriptFound

WINDOW_LENGTH = 7


@st.cache_resource
def init():
    summarizer = (
        Summarizer.get_singleton()
    )  # Summarizer for generating concise summaries
    model = SentenceTransformer(
        "msmarco-distilbert-base-dot-prod-v3"
    )  # SentenceTransformer model for semantic search
    return summarizer, model


# Function to summarize text
def summarize(text):
    return summarizer.summarize(
        text
    )  # Generate summary using the initialized summarizer


# Function to download subtitles from a YouTube video
@st.cache_data
def download_subtitles(url):
    video_id = urlparse(url).query[2::]  # Extract video ID from URL

    try:
        # Attempt to fetch English transcript directly
        return YouTubeTranscriptApi.get_transcript(video_id)
    except NoTranscriptFound:
        # Handle case where English transcript is not available
        st.error(
            "English transcript not found. Searching for available translations..."
        )

        # Attempt to fetch and translate available transcripts
        for transcript in YouTubeTranscriptApi.list_transcripts(video_id):
            try:
                return transcript.translate("en").fetch()
            except Exception as e:
                st.warning(f"Error translating transcript: {e}")

        st.error("No usable transcripts found.")
        return None


# Function to parse subtitles into a DataFrame
@st.cache_data
def parse_subtitles(url):
    subtitles = download_subtitles(url)
    if subtitles:
        return pd.DataFrame(subtitles)
    else:
        return None


# Function to summarize subtitles
@st.cache_data
def get_summary(url):
    subtitles = download_subtitles(url)
    text_st = " ".join(pd.DataFrame(subtitles)["text"])
    try:
        return summarize(text_st)
    except BaseException as e:
        print(e)
        traceback.print_exc()
        return e


# Function to create embeddings for semantic search
def store_embeddings(subtitle_df):
    embeddings = model.encode(subtitle_df["text"].tolist())
    index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
    index.add_with_ids(embeddings.astype("float32"), np.arange(len(subtitle_df)))
    return index


# Function to perform semantic search within subtitles
def search(subtitle_df, query, top_k, index):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    result_df = subtitle_df.iloc[indices.tolist()[0]].copy()
    result_df["distance"] = distances.tolist()[0]
    return result_df


# Function to retrieve relevant lines based on a search phrase
@st.cache_data
def get_relevant_lines(subtitle_df, search_phrase, window_length):
    df = subtitle_df.copy()
    df["text"] = df["text"].astype(str)
    df["start"] = (df["start"] // window_length) * window_length
    df = df.groupby("start").agg({"text": " ".join}).reset_index()
    index = store_embeddings(df)
    return search(df, search_phrase, 6, index)


# --------------------------------------------------
# Main application logic
# --------------------------------------------------

if __name__ == "__main__":
    try:
        # Initialize required components
        summarizer, model = init()

        with streamlit_analytics.track(unsafe_password="credict123"):
            # Set page title and description
            st.title("Bert-Video-Search-and-Jump")
            st.write(
                "**An AI-based tool to semantic search through YouTube video subtitles and jump to relevant sections**"
            )

            # Prompt for YouTube video URL
            vid_url = st.text_input(
                "YouTube Video URL", placeholder="Enter YouTube video url here"
            )

            if vid_url:
                # Display video player
                vid_placeholder = st.empty()
                with vid_placeholder.container():
                    st_player(vid_url, playing=True)

                # Prompt for search phrase
                searchphrase = st.text_input(
                    "Enter Search keywords here relevant to the topic you are searching for in this video"
                )

                context_window_duration = st.number_input(
                    "Enter context window duration in seconds", min_value=1, value=10
                )

                # Handle subtitle parsing and analysis
                analysis_placeholder = st.empty()  # Container for analysis results
                analysis_placeholder.empty()
                subtitle_df = parse_subtitles(vid_url)  # Parse subtitles
                subtitle_df.to_csv("subtitles.csv")  # Save subtitles to CSV

                if searchphrase:
                    print("\n\n\n Searching", searchphrase)
                    search_results = get_relevant_lines(
                        subtitle_df, searchphrase, context_window_duration
                    )

                    with analysis_placeholder.container():
                        if len(search_results):
                            st.text("Relevant sections below: ")
                            for cap, start in zip(
                                search_results["text"].to_list(),
                                search_results["start"].to_list(),
                            ):
                                col1, col2 = st.columns([1, 4])
                                col1.button(
                                    "Jump to ",
                                    key=" ".join(
                                        [
                                            "Jump",
                                            vid_url,
                                            str(start),
                                            str(random.randint(0, 9999999)),
                                            cap,
                                        ]
                                    ),
                                )
                                col2.markdown(cap)
                        else:
                            st.text("No relevant section found, try something else ...")

                # Handle "Jump" button clicks
                for k, v in st.session_state.items():
                    if k.startswith("Jump") and v is True:
                        _, new_url, start, _ = k.split(maxsplit=3)
                        vid_placeholder.empty()
                        with vid_placeholder.container():
                            st_player(
                                vid_url + "&t={}s".format(round(float(start))),
                                playing=True,
                            )

                # Generate and display video summary
                summary = get_summary(vid_url)
                if summary:
                    st.subheader("Summary of the video: \n\n")
                    st.markdown(summary)

    except BaseException as e:
        print(e)
        traceback.print_exc()
        st.text("Some error occurred, please ensure YouTube URL is correct")
