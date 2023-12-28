---
title: Bert Video Search And Jump
emoji: 📚
colorFrom: blue
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

Check Out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Bert-Video-Search-and-Jump

**An AI-powered web app to search and navigate YouTube videos with ease.**

## Demo

Demo link: https://huggingface.co/spaces/Instantaneous1/bert-video-search-and-jump

![Demo](ui.png)

## Key Features

- **Downloads subtitles** from YouTube videos.
- **Enables semantic search** within videos using subtitle embeddings.
- **Provides concise summaries** of entire videos.

## Technologies

- Python
- Streamlit
- Streamlit Analytics
- Natural language processing (NLP) techniques

## Usage

1. Clone this repository.
2. Install required dependencies: `pip install -r requirements.txt`
3. Save your gemini secrets in `.streamlit/secrets.toml` file as `GOOGLE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"`
4. Run the app: `streamlit run app.py`
