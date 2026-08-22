import numpy as np
import pandas as pd


def obter_coluna(df, opcoes):
    for coluna in opcoes:
        if coluna in df.columns:
            return coluna
    return None


def normalizar_tempo(dias):
    return min(max(float(dias) / 30, 0), 1)


def normalizar_exposicoes(exposicoes):
    return 1 - min(max(float(exposicoes) / 10, 0), 1)


def calcular_indice_prioridade(
    recall,
    dias_sem_pratica,
    exposicoes,
    dificuldade
):
    recall = min(max(float(recall), 0), 1)
    dificuldade = min(max(float(dificuldade), 0), 1)

    risco_recall = 1 - recall
    fator_tempo = normalizar_tempo(dias_sem_pratica)
    fator_exposicoes = normalizar_exposicoes(exposicoes)

    indice = (
        0.40 * risco_recall
        + 0.30 * fator_tempo
        + 0.15 * fator_exposicoes
        + 0.15 * dificuldade
    ) * 100

    return round(min(max(indice, 0), 100), 1)


def classificar_prioridade(indice):
    if indice <= 30:
        return "Baixa"
    if indice <= 60:
        return "Média"
    if indice <= 80:
        return "Alta"
    return "Crítica"


def obter_recomendacao(prioridade):
    recomendacoes = {
        "Baixa": (
            "Não há necessidade de revisão imediata. "
            "O conteúdo pode permanecer no intervalo normal de estudo."
        ),
        "Média": (
            "Acompanhar a retenção e programar uma nova revisão "
            "conforme a rotina de estudo."
        ),
        "Alta": (
            "Priorizar a revisão deste conteúdo. "
            "O cenário indica risco significativo de esquecimento."
        ),
        "Crítica": (
            "Realizar a revisão o quanto antes. "
            "O conteúdo apresenta prioridade crítica."
        )
    }
    return recomendacoes.get(
        prioridade,
        "Não foi possível gerar uma recomendação."
    )


def estimar_recall_cenario(
    curve,
    idioma,
    dias_sem_pratica,
    exposicoes,
    recall_historico=0.85
):
    if curve.empty:
        return recall_historico

    df = curve.copy()

    if idioma != "Todos" and "idioma" in df.columns:
        filtrado = df[df["idioma"].astype(str) == str(idioma)]
        if not filtrado.empty:
            df = filtrado

    recall_col = obter_coluna(
        df,
        ["item_recall_rate", "avg_session_recall", "recall"]
    )
    lag_col = obter_coluna(
        df,
        ["lag_days", "avg_lag_days"]
    )
    exposure_col = obter_coluna(
        df,
        ["avg_prior_exposures", "prior_exposures", "exposures"]
    )

    if recall_col is None:
        return recall_historico

    if lag_col is None or exposure_col is None:
        valor = df[recall_col].mean()
        if pd.isna(valor):
            return recall_historico
        return float(valor)

    dados = df[[lag_col, exposure_col, recall_col]].dropna()
    if dados.empty:
        return recall_historico

    lag_max = max(float(dados[lag_col].max()), 1)
    exposure_max = max(float(dados[exposure_col].max()), 1)

    dados = dados.copy()
    dados["distancia"] = (
        abs(dados[lag_col] - dias_sem_pratica) / lag_max
        + abs(dados[exposure_col] - exposicoes) / exposure_max
    )

    linha = dados.loc[dados["distancia"].idxmin()]
    recall_base = float(linha[recall_col])

    recall_estimado = (
        0.80 * recall_base
        + 0.20 * recall_historico
    )

    return min(max(recall_estimado, 0), 1)


def calcular_dificuldade_palavra(
    words,
    idioma="Todos",
    palavra=None
):
    if words.empty:
        return 0.5, "Média geral"

    df = words.copy()

    if idioma != "Todos" and "idioma" in df.columns:
        filtrado = df[df["idioma"].astype(str) == str(idioma)]
        if not filtrado.empty:
            df = filtrado

    recall_col = obter_coluna(
        df,
        ["item_recall_rate", "avg_recall", "recall"]
    )

    if recall_col is None:
        return 0.5, "Média geral"

    palavra_col = obter_coluna(
        df,
        ["surface_form", "lemma", "word"]
    )

    if palavra is not None and palavra != "Automática":
        if palavra_col is not None:
            resultado = df[
                df[palavra_col].astype(str) == str(palavra)
            ]
            if not resultado.empty:
                recall = resultado[recall_col].mean()
                dificuldade = 1 - recall
                return (
                    float(min(max(dificuldade, 0), 1)),
                    str(palavra)
                )

    recall_medio = df[recall_col].mean()
    if pd.isna(recall_medio):
        return 0.5, "Média geral"

    dificuldade = 1 - recall_medio
    return (
        float(min(max(dificuldade, 0), 1)),
        "Média do idioma"
    )


def calcular_prioridades_conteudos(
    words,
    tempo_padrao=14,
    exposicoes_padrao=5
):
    if words.empty:
        return pd.DataFrame()

    df = words.copy()

    recall_col = obter_coluna(
        df,
        ["item_recall_rate", "avg_recall", "recall"]
    )

    if recall_col is None:
        return pd.DataFrame()

    df["recall_utilizado"] = (
        pd.to_numeric(df[recall_col], errors="coerce")
        .fillna(pd.to_numeric(df[recall_col], errors="coerce").mean())
    )

    df["dificuldade"] = 1 - df["recall_utilizado"]

    df["indice_prioridade"] = df.apply(
        lambda linha: calcular_indice_prioridade(
            recall=linha["recall_utilizado"],
            dias_sem_pratica=tempo_padrao,
            exposicoes=exposicoes_padrao,
            dificuldade=linha["dificuldade"]
        ),
        axis=1
    )

    df["prioridade"] = df["indice_prioridade"].apply(
        classificar_prioridade
    )

    return df.sort_values(
        "indice_prioridade",
        ascending=False
    )