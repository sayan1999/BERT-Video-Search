import pandas as pd

df = pd.read_csv("subtitles.csv")

window_length = 60
df["start"] = df["start"].astype(float)
df["text"] = df["text"].astype(str)


df["start"] = (df["start"] // 60) * 60
print(df.groupby("start").agg({"text": " ".join}).reset_index())
