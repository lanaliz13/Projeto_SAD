import numpy as np
import pandas as pd


def obter_coluna(df, opcoes):
    if df is None or df.empty:
        return None
    for coluna in opcoes:
        if coluna in df.columns:
            return coluna
    return None


def normalizar_tempo(dias):
    dias = float(dias)

    if dias <= 0:
        return 0.0

    if dias < 1:
        return 0.10

    if dias < 3:
        return 0.20

    if dias < 7:
        return 0.35

    if dias < 14:
        return 0.50

    if dias < 30:
        return 0.65

    if dias < 90:
        return 0.80

    return 1.0


def normalizar_exposicoes(exposicoes):
    exposicoes = float(exposicoes)

    if exposicoes <= 0:
        return 1.0

    if exposicoes <= 2:
        return 0.80

    if exposicoes <= 4:
        return 0.60

    if exposicoes <= 9:
        return 0.40

    if exposicoes <= 19:
        return 0.20

    return 0.0


def calcular_indice_prioridade(
    recall,
    dias_sem_pratica,
    exposicoes,
    dificuldade
):
    recall = min(max(float(recall), 0.0), 1.0)
    dificuldade = min(max(float(dificuldade), 0.0), 1.0)

    risco_recall = 1.0 - recall
    fator_tempo = normalizar_tempo(dias_sem_pratica)
    fator_exposicoes = normalizar_exposicoes(exposicoes)

    # Recalibragem de pesos para dar maior sensibilidade ao risco e tempo sem prática
    indice = (
        0.55 * risco_recall
        + 0.25 * fator_tempo
        + 0.10 * fator_exposicoes
        + 0.10 * dificuldade
    ) * 100.0

    return round(min(max(indice, 0.0), 100.0), 1)


def classificar_prioridade(indice):
    if indice <= 30.0:
        return "Baixa"
    if indice <= 50.0:
        return "Média"
    if indice <= 70.0:
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
            "O conteúdo apresenta prioridade crítica e alto risco de esquecimento."
        )
    }
    return recomendacoes.get(
        prioridade,
        "Não foi possível gerar uma recomendação."
    )


def classificar_faixa_tempo(dias):
    dias = float(dias)

    if dias < 1 / 24:
        return "<1 hour"
    elif dias < 6 / 24:
        return "1-6 hours"
    elif dias < 1:
        return "6-24 hours"
    elif dias < 3:
        return "1-3 days"
    elif dias < 7:
        return "3-7 days"
    elif dias < 14:
        return "1-2 weeks"
    elif dias < 30:
        return "2-4 weeks"
    elif dias < 90:
        return "1-3 months"
    else:
        return "3+ months"


def classificar_faixa_exposicoes(exposicoes):
    exposicoes = int(exposicoes)

    if exposicoes <= 2:
        return "1-2 exposures"
    elif exposicoes <= 4:
        return "3-4 exposures"
    elif exposicoes <= 9:
        return "5-9 exposures"
    elif exposicoes <= 19:
        return "10-19 exposures"
    else:
        return "20+ exposures"


def estimar_recall_cenario(
    curve,
    idioma,
    dias_sem_pratica,
    exposicoes,
    recall_historico=0.85
):
    if curve is None or curve.empty:
        return recall_historico

    df = curve.copy()

    # ---------------------------------------------------------
    # FILTRO POR IDIOMA
    # ---------------------------------------------------------

    if (
        idioma != "Todos"
        and "idioma" in df.columns
    ):
        filtrado = df[
            df["idioma"].astype(str).str.lower()
            == str(idioma).lower()
        ]

        if not filtrado.empty:
            df = filtrado

    # ---------------------------------------------------------
    # IDENTIFICA COLUNAS DA CURVA
    # ---------------------------------------------------------

    lag_col = obter_coluna(
        df,
        ["lag_bin", "lag_days", "avg_lag_days"]
    )

    exp_col = obter_coluna(
        df,
        [
            "practice_bin",
            "avg_prior_exposures",
            "prior_exposures",
            "exposures"
        ]
    )

    recall_col = obter_coluna(
        df,
        [
            "avg_session_recall",
            "item_recall_rate",
            "avg_recall",
            "recall"
        ]
    )

    if recall_col is None:
        return recall_historico

    # ---------------------------------------------------------
    # CASO A CURVA POSSUA AS FAIXAS DO DATASET
    # ---------------------------------------------------------

    if (
        lag_col == "lag_bin"
        and exp_col == "practice_bin"
    ):

        faixa_tempo = classificar_faixa_tempo(
            dias_sem_pratica
        )

        faixa_exposicoes = classificar_faixa_exposicoes(
            exposicoes
        )

        resultado = df[
            (df[lag_col].astype(str) == faixa_tempo)
            &
            (df[exp_col].astype(str) == faixa_exposicoes)
        ]

        if not resultado.empty:

            recall = resultado[recall_col].mean()

            if not pd.isna(recall):
                return float(
                    min(
                        max(recall, 0.15),
                        0.98
                    )
                )

        # -----------------------------------------------------
        # FALLBACK: MESMA FAIXA DE TEMPO
        # -----------------------------------------------------

        resultado_tempo = df[
            df[lag_col].astype(str) == faixa_tempo
        ]

        if not resultado_tempo.empty:

            recall = resultado_tempo[recall_col].mean()

            if not pd.isna(recall):
                return float(
                    min(
                        max(recall, 0.15),
                        0.98
                    )
                )

    # ---------------------------------------------------------
    # CASO TENHA COLUNAS NUMÉRICAS
    # ---------------------------------------------------------

    if lag_col and exp_col:

        dados = df[
            [lag_col, exp_col, recall_col]
        ].copy()

        dados[lag_col] = pd.to_numeric(
            dados[lag_col],
            errors="coerce"
        )

        dados[exp_col] = pd.to_numeric(
            dados[exp_col],
            errors="coerce"
        )

        dados[recall_col] = pd.to_numeric(
            dados[recall_col],
            errors="coerce"
        )

        dados = dados.dropna()

        if not dados.empty:

            lag_max = max(
                float(dados[lag_col].max()),
                1.0
            )

            exp_max = max(
                float(dados[exp_col].max()),
                1.0
            )

            dados["distancia"] = (
                abs(
                    dados[lag_col]
                    - dias_sem_pratica
                ) / lag_max
                +
                abs(
                    dados[exp_col]
                    - exposicoes
                ) / exp_max
            )

            linha = dados.loc[
                dados["distancia"].idxmin()
            ]

            return float(
                min(
                    max(
                        float(linha[recall_col]),
                        0.15
                    ),
                    0.98
                )
            )

    # ---------------------------------------------------------
    # ÚLTIMO FALLBACK
    # ---------------------------------------------------------

    valor = df[recall_col].mean()

    if pd.isna(valor):
        return recall_historico

    return float(
        min(
            max(float(valor), 0.15),
            0.98
        )
    )


def calcular_dificuldade_palavra(
    words,
    idioma="Todos",
    palavra=None
):
    if words is None or words.empty:
        return 0.5, "Média geral"

    df = words.copy()

    if idioma != "Todos" and "idioma" in df.columns:
        filtrado = df[df["idioma"].astype(str).str.lower() == str(idioma).lower()]
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
                dificuldade = 1.0 - recall
                return (
                    float(min(max(dificuldade, 0.0), 1.0)),
                    str(palavra)
                )

    recall_medio = df[recall_col].mean()
    if pd.isna(recall_medio):
        return 0.5, "Média geral"

    dificuldade = 1.0 - recall_medio
    return (
        float(min(max(dificuldade, 0.0), 1.0)),
        "Média do idioma"
    )


def calcular_prioridades_conteudos(
    words,
    tempo_padrao=14,
    exposicoes_padrao=5
):
    if words is None or words.empty:
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

    df["dificuldade"] = 1.0 - df["recall_utilizado"]

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