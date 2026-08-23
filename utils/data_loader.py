import os
import pandas as pd
import streamlit as st


def obter_coluna(df, opcoes):
    if df is None or df.empty:
        return None
    for coluna in opcoes:
        if coluna in df.columns:
            return coluna
    return None


def padronizar_idiomas(df):
    if df is None or df.empty:
        return df

    # Mapeamento para garantir consistência entre códigos ('de') e nomes ('German')
    mapa_codigos = {
        'de': 'German',
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'it': 'Italian',
        'pt': 'Portuguese',
        'all': 'All languages'
    }

    # 1. Tenta pegar coluna com nome completo
    col_nome = obter_coluna(df, ['language_name', 'learning_language_name', 'ui_language_name'])
    if col_nome:
        df['idioma'] = df[col_nome].astype(str)
        return df

    # 2. Tenta pegar coluna com código ISO
    col_codigo = obter_coluna(df, ['learning_language', 'language', 'ui_language'])
    if col_codigo:
        df['idioma'] = df[col_codigo].astype(str).map(lambda x: mapa_codigos.get(x, x))
        return df

    df['idioma'] = 'Todos'
    return df


@st.cache_data
def carregar_dados():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    # Tratamento de diretório flexível
    if not os.path.exists(data_dir):
        data_dir = base_dir

    caminhos = {
        "traces": os.path.join(data_dir, "learning_traces_sample.csv"),
        "courses": os.path.join(data_dir, "language_courses.csv"),
        "curve": os.path.join(data_dir, "forgetting_curve.csv"),
        "words": os.path.join(data_dir, "word_difficulty.csv")
    }

    dados = {}
    for nome, caminho in caminhos.items():
        if os.path.exists(caminho):
            try:
                dados[nome] = pd.read_csv(caminho)
            except Exception:
                dados[nome] = pd.DataFrame()
        else:
            dados[nome] = pd.DataFrame()

    traces = padronizar_idiomas(dados["traces"])
    courses = padronizar_idiomas(dados["courses"])
    curve = padronizar_idiomas(dados["curve"])
    words = padronizar_idiomas(dados["words"])

    if not traces.empty and "practice_time" in traces.columns:
        traces["practice_time"] = pd.to_datetime(traces["practice_time"], errors="coerce")

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
        "cnjcoo": "Conjunção",
        "cnjsub": "Conjunção"
    }

    if not words.empty and "pos" in words.columns:
        words["classe_gramatical"] = words["pos"].map(pos_map).fillna(words["pos"]).fillna("Outros")

    return {
        "traces": traces,
        "courses": courses,
        "curve": curve,
        "words": words
    }