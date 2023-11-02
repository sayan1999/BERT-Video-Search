import random, re
from urllib.parse import urlparse
from yt_dlp import YoutubeDL
import glob
import webvtt
import pandas as pd
from streamlit_player import st_player
from youtube_transcript_api import YouTubeTranscriptApi


from scipy import spatial
from gensim.models import word2vec

from collections import namedtuple
import nltk
import pandas as pd
import gensim
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import streamlit as st


@st.cache_data
def dl_transcript(url):
    url_data = urlparse(url)
    print("id", url_data.query[2::])
    return YouTubeTranscriptApi.get_transcript(url_data.query[2::])


@st.cache_data
def init():
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    model = gensim.models.KeyedVectors.load_word2vec_format(
        "archive/GoogleNews-vectors-negative300-SLIM.bin",
        binary=True,
    )
    # model = None
    return tokenizer, model


# @st.cache_data
def docsimilarity(model, keyword, doc):
    cutoff = 0.4
    score = 0
    for w in doc:
        sm = model.similarity(keyword, w) if w in model else 0
        if sm >= cutoff:
            score += sm
    return score


@st.cache_data
def get_relevant_line(df, searchphrase):
    tokenizer, model = init()
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    df = preprocess(df, tokenizer, wordnet_lemmatizer, stop_words)

    keywords = [
        wordnet_lemmatizer.lemmatize(
            wordnet_lemmatizer.lemmatize(
                wordnet_lemmatizer.lemmatize(kw.lower()), pos="v"
            ),
            pos=("a"),
        )
        for kw in tokenizer.tokenize(searchphrase)
    ]
    print("lemm keywords: ", keywords)
    df["similarity"] = sum(
        [
            df["docs"].apply(lambda doc: docsimilarity(model, keyword.lower(), doc))
            for keyword in keywords
            if keyword in model
        ]
    )
    df["docs"] = df["docs"].apply(" ".join)
    df = df.sort_values("similarity", ascending=False)
    df.to_csv("result.csv", index=False)
    res_idx = df["similarity"] >= 1
    print(
        "Result length: ",
        sum(res_idx),
    )
    return df[res_idx].reset_index().iloc[:4]


@st.cache_data
def parse_subtitles(url):
    return pd.DataFrame(dl_transcript(url))


# @st.cache_data
def preprocess(df, tokenizer, wordnet_lemmatizer, stop_words):
    orig_docs = [[word for word in tokenizer.tokenize(sent)] for sent in df["text"]]

    df["docs"] = [
        [
            wordnet_lemmatizer.lemmatize(
                wordnet_lemmatizer.lemmatize(
                    wordnet_lemmatizer.lemmatize(word.lower()), pos="v"
                ),
                pos=("a"),
            )
            for word in sent
            if word not in stop_words
        ]
        for sent in orig_docs
    ]
    # print(df["docs"])
    return df


def vidattstamp(vid_url, start, vid_placeholder):
    vid_url = vid_url + "&t=400s"
    print("Skipping to ", start, vid_url)
    vid_placeholder.empty()
    # with placeholder.container():
    #     st_player(vid_url, playing=True, muted=True)


vid_url = st.text_input("Youtube video")
if vid_url:
    # print(st.session_state)
    placeholder = st.empty()
    analysis_placeholder = st.empty()
    with placeholder.container():
        st_player(vid_url, playing=True)
        analysis_placeholder.empty()
        # st.video(vid_url)
    df = parse_subtitles(vid_url)
    df.to_csv("caps.csv")
    searchphrase = st.text_input(
        "Search keywords relevant to section you are searching for in this video"
    )
    if searchphrase:
        print("\n\n\n Searching", searchphrase)
        df = get_relevant_line(df, searchphrase)
        # print(df)
        with analysis_placeholder.container():
            if len(df):
                st.text("Relevant sections below: ")
                # placeholder.empty()
                # st.dataframe(df)
                for cap, start in zip(df["text"].to_list(), df["start"].to_list()):
                    col1, col2 = st.columns([1, 4])
                    col1.button(
                        "Jump to time: " + str(start),
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
            placeholder.empty()
            with placeholder.container():
                st_player(vid_url + "&t={}s".format(round(float(start))), playing=True)
