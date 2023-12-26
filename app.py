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


# def set_page_config():
#     """Sets the page configuration."""
#     st.set_page_config(
#         page_title="Sephora vs Ulta",
#         layout="wide",
#     )


# set_page_config()


@st.cache_resource
def init():
    CLIENT = Summarizer()
    MODEL = SentenceTransformer("msmarco-distilbert-base-dot-prod-v3")
    return CLIENT, MODEL


def summarize(txt):
    global CLIENT
    print(">>>>>>", CLIENT)
    return CLIENT.summarize(txt)


@st.cache_data
def dl_subtitle(url):
    url_data = urlparse(url)
    print("Id:", url_data.query[2::])
    try:
        return YouTubeTranscriptApi.get_transcript(url_data.query[2::])
    except BaseException as e:
        print(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\nCould not find english transcript, looking for translation",
        )
        for t in YouTubeTranscriptApi.list_transcripts(url_data.query[2::]):
            return t.translate("en").fetch()


@st.cache_data
def get_summary(url):
    subtitles = dl_subtitle(url)
    text_st = " ".join(pd.DataFrame(subtitles)["text"])
    try:
        return summarize(text_st)
    except BaseException as e:
        print(e)
        traceback.print_exc()
        return e


@st.cache_data
def parse_subtitles(url):
    subtitles = dl_subtitle(url)
    return pd.DataFrame(subtitles)


def store_embeddings(subtitle_df):
    encoded_data = MODEL.encode(subtitle_df.text.tolist())
    encoded_data = np.asarray(encoded_data.astype("float32"))
    index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
    index.add_with_ids(encoded_data, np.array(range(0, len(subtitle_df))))
    return index


def search(subtitle_df, query, top_k, index):
    query_vector = MODEL.encode([query])
    top_k = index.search(query_vector, top_k)
    distances = top_k[0].tolist()[0]
    top_k_ids = top_k[1].tolist()[0]
    top_k_ids = list(np.unique(top_k_ids))
    resultdf = subtitle_df.iloc[top_k_ids].copy()
    resultdf["dist"] = distances
    print(distances)
    return resultdf


@st.cache_data
def get_relevant_line(subtitle_df, searchphrase):
    index = store_embeddings(subtitle_df)
    return search(subtitle_df, searchphrase, 6, index)


if __name__ == "__main__":
    try:
        CLIENT, MODEL = init()
        with streamlit_analytics.track(unsafe_password="credict123"):
            # st.set_page_config(page_title="Bert-Video-Search-and-Jump")
            st.title("Bert-Video-Search-and-Jump")
            st.write(
                "**An AI based tool to semantic search through an Youtube video subtitles and jump to relevant sections**"
            )
            vid_url = st.text_input(
                "Youtube Video URL", placeholder="Enter Youtube video url here"
            )
            if vid_url:
                vid_placeholder = st.empty()

                with vid_placeholder.container():
                    st_player(vid_url, playing=True)
                searchphrase = st.text_input(
                    "Enter Search keywords here relevant to the topic you are searching for in this video"
                )
                analysis_placeholder = st.empty()
                analysis_placeholder.empty()
                subtitle_df = parse_subtitles(vid_url)
                subtitle_df.to_csv("subtitles.csv")

                if searchphrase:
                    print("\n\n\n Searching", searchphrase)
                    search_results = get_relevant_line(subtitle_df, searchphrase)
                    # print(df)
                    with analysis_placeholder.container():
                        if len(search_results):
                            st.text("Relevant sections below: ")
                            for cap, start in zip(
                                search_results["text"].to_list(),
                                search_results["start"].to_list(),
                            ):
                                col1, col2 = st.columns([1, 4])
                                col1.button(
                                    "Jump to section ",
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

                for k, v in st.session_state.items():
                    if k.startswith("Jump") and v is True:
                        print(k.split(maxsplit=3))
                        _, new_url, start, _ = k.split(maxsplit=3)
                        vid_placeholder.empty()
                        with vid_placeholder.container():
                            st_player(
                                vid_url + "&t={}s".format(round(float(start))),
                                playing=True,
                            )

                summary = get_summary(vid_url)
                if summary:
                    st.subheader("Summary of the video: \n\n")
                    st.markdown(summary)
    except BaseException as e:
        print(e)
        traceback.print_exc()

        st.text("Some error occured, please ensure youtube URL is correct")
