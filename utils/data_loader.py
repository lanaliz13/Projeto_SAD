import os
import pandas as pd
import streamlit as st


def obter_coluna(df, opcoes):
    for coluna in opcoes:
        if coluna in df.columns:
            return coluna
    return None


def adicionar_coluna_idioma(df):
    coluna_idioma = obter_coluna(
        df,
        [
            "learning_language_name",
            "learning_language",
            "language_name",
            "language"
        ]
    )

    if coluna_idioma is not None:
        df["idioma"] = df[coluna_idioma].astype(str)

    return df


@st.cache_data
def carregar_dados():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    caminhos = {
        "traces": os.path.join(data_dir, "learning_traces_sample.csv"),
        "courses": os.path.join(data_dir, "language_courses.csv"),
        "curve": os.path.join(data_dir, "forgetting_curve.csv"),
        "words": os.path.join(data_dir, "word_difficulty.csv")
    }

    dados = {}

    for nome, caminho in caminhos.items():
        if os.path.exists(caminho):
            dados[nome] = pd.read_csv(caminho)
        else:
            dados[nome] = pd.DataFrame()

    traces = adicionar_coluna_idioma(dados["traces"])
    courses = adicionar_coluna_idioma(dados["courses"])
    curve = adicionar_coluna_idioma(dados["curve"])
    words = adicionar_coluna_idioma(dados["words"])

    if "practice_time" in traces.columns:
        traces["practice_time"] = pd.to_datetime(
            traces["practice_time"],
            errors="coerce"
        )

    pos_map = {
        "vblex": "Verbo",
        "vbser": "Verbo",
        "vbhaver": "Verbo",
        "n": "Substantivo",
        "np": "Nome próprio",
        "adj": "Adjetivo",
        "adv": "Advérbio",
        "pr": "Preposição",
        "det": "Determinante",
        "prn": "Pronome",
        "num": "Numeral",
        "cnjcoo": "Conjunção coordenativa",
        "cnjsub": "Conjunção subordinativa"
    }

    if "pos" in words.columns:
        words["classe_gramatical"] = (
            words["pos"]
            .map(pos_map)
            .fillna(words["pos"])
            .fillna("Não informada")
        )

    return {
        "traces": traces,
        "courses": courses,
        "curve": curve,
        "words": words
    }