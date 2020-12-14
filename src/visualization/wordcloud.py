import pandas as pd
from wordcloud import WordCloud
from typing import Set
from src.data.utils import get_frequencies
import plotly.express as px


def get_wordcloud(series: pd.Series, stop_words: Set[str] = None) -> WordCloud:
    frequencies = get_frequencies(series, stop_words)
    wordcloud = WordCloud(
        width=1920, height=1080, background_color="white"
    ).generate_from_frequencies(frequencies=frequencies)

    return wordcloud


def plot_wordcloud(series: pd.Series, stop_words: Set[str] = None):
    wordcloud = get_wordcloud(series, stop_words)
    fig = px.imshow(wordcloud)
    fig.show()