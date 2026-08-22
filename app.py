import os
import html
import textwrap
import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import carregar_dados, obter_coluna
from utils.metrics import (
    estimar_recall_cenario,
    calcular_dificuldade_palavra,
    calcular_indice_prioridade,
    classificar_prioridade,
    obter_recomendacao,
    calcular_prioridades_conteudos
)


st.set_page_config(
    page_title="Duolingo Learning Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)


def render_html(content, unsafe_allow_html=True):
    """
    Renderiza HTML diretamente no Streamlit.
    Evita que tags <div>, <style>, etc. sejam
    interpretadas como bloco de código Markdown.
    """
    conteudo = textwrap.dedent(str(content)).strip()

    if hasattr(st, "html"):
        st.html(conteudo)
    else:
        st.markdown(
            conteudo,
            unsafe_allow_html=unsafe_allow_html
        )


def carregar_css():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(base_dir, "assets", "style.css")

    if not os.path.exists(caminho):
        st.warning(f"Arquivo CSS não encontrado: {caminho}")
        return

    with open(caminho, "r", encoding="utf-8") as arquivo:
        css = arquivo.read()

    if hasattr(st, "html"):
        st.html(f"<style>{css}</style>")
    else:
        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True
        )


def percentual(valor):
    if pd.isna(valor):
        return "Não disponível"

    return f"{float(valor) * 100:.1f}%"


def numero(valor):
    if pd.isna(valor):
        return "0"

    return f"{int(valor):,}".replace(",", ".")


def valor_numerico(valor, casas=1):
    if pd.isna(valor):
        return "Não disponível"

    return f"{float(valor):.{casas}f}"


def cabecalho(titulo, descricao):
    render_html(
        f"""
        <div class="app-header">
            <div class="app-eyebrow">SISTEMA DE APOIO À DECISÃO</div>
            <h1>{html.escape(titulo)}</h1>
            <p>{html.escape(descricao)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def secao(titulo, descricao=""):
    render_html(
        f"""
        <div class="section-title">
            <h2>{html.escape(titulo)}</h2>
            <p>{html.escape(descricao)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def caixa_metrica(titulo, valor, descricao):
    render_html(
        f"""
        <div class="metric-box">
            <div class="metric-label">{html.escape(str(titulo))}</div>
            <div class="metric-value">{html.escape(str(valor))}</div>
            <div class="metric-description">{html.escape(str(descricao))}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def filtrar_idioma(df, idioma):
    if df.empty or idioma == "Todos":
        return df.copy()

    if "idioma" not in df.columns:
        return df.copy()

    resultado = df[
        df["idioma"].astype(str) == str(idioma)
    ].copy()

    if resultado.empty:
        return df.copy()

    return resultado


def obter_idiomas(*dataframes):
    idiomas = set()

    for df in dataframes:
        if not df.empty and "idioma" in df.columns:
            valores = (
                df["idioma"]
                .dropna()
                .astype(str)
                .unique()
            )

            idiomas.update(valores)

    return sorted(idiomas)


def obter_nome_palavra(df):
    return obter_coluna(
        df,
        [
            "surface_form",
            "lemma",
            "word"
        ]
    )


def classe_status(prioridade):
    mapa = {
        "Baixa": "status-low",
        "Média": "status-medium",
        "Alta": "status-high",
        "Crítica": "status-critical"
    }

    return mapa.get(prioridade, "status-medium")


def normalizar_percentual(valor):
    if pd.isna(valor):
        return 0

    valor = float(valor)

    if valor > 1:
        valor = valor / 100

    return max(0, min(valor, 1))


def barra_horizontal(valor, maximo=100, classe="bar-fill"):
    if pd.isna(valor):
        percentual_barra = 0
    else:
        percentual_barra = (
            float(valor) / float(maximo) * 100
            if maximo != 0
            else 0
        )

    percentual_barra = max(0, min(percentual_barra, 100))

    return f"""
    <div class="mini-bar">
        <div class="{classe}" style="width:{percentual_barra:.2f}%"></div>
    </div>
    """


def tabela_html(df, colunas=None, max_linhas=15, formatos=None):
    if df is None or df.empty:
        render_html(
            """
            <div class="empty-state">
                Nenhum resultado foi encontrado com os filtros selecionados.
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    if colunas is None:
        colunas = list(df.columns)

    colunas = [col for col in colunas if col in df.columns]

    if not colunas:
        render_html(
            """
            <div class="empty-state">
                Não existem colunas disponíveis para exibição.
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    dados = df[colunas].head(max_linhas).copy()

    tabela = """
    <div class="custom-table-wrapper">
        <table class="custom-table">
            <thead>
                <tr>
    """

    for coluna in colunas:
        tabela += f"<th>{html.escape(str(coluna))}</th>"

    tabela += """
                </tr>
            </thead>
            <tbody>
    """

    formatos = formatos or {}

    for _, linha in dados.iterrows():
        tabela += "<tr>"

        for coluna in colunas:
            valor = linha[coluna]

            if coluna in formatos and pd.notna(valor):
                try:
                    valor_formatado = formatos[coluna](valor)
                except Exception:
                    valor_formatado = str(valor)
            elif pd.isna(valor):
                valor_formatado = "—"
            elif isinstance(valor, (float, np.floating)):
                valor_formatado = f"{valor:.2f}"
            else:
                valor_formatado = str(valor)

            tabela += f"<td>{html.escape(valor_formatado)}</td>"

        tabela += "</tr>"

    tabela += """
            </tbody>
        </table>
    </div>
    """

    render_html(tabela, unsafe_allow_html=True)


def ranking_cards(
    df,
    titulo_col,
    valor_col,
    max_linhas=10,
    percentual_valor=False,
    subtitulo_col=None
):
    if df.empty or titulo_col not in df.columns or valor_col not in df.columns:
        render_html(
            """
            <div class="empty-state">
                Não foi possível montar o ranking com as colunas disponíveis.
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    dados = df.head(max_linhas).copy()

    valores = pd.to_numeric(
        dados[valor_col],
        errors="coerce"
    ).fillna(0)

    if percentual_valor:
        valores_barra = valores * 100
        maximo = 100
    else:
        valores_barra = valores
        maximo = valores_barra.max()

        if maximo <= 0:
            maximo = 1

    for indice, (_, linha) in enumerate(dados.iterrows(), start=1):
        titulo = str(linha[titulo_col])

        valor = pd.to_numeric(
            pd.Series([linha[valor_col]]),
            errors="coerce"
        ).iloc[0]

        if pd.isna(valor):
            valor = 0

        if percentual_valor:
            texto_valor = f"{valor * 100:.1f}%"
            largura = max(0, min(valor * 100, 100))
        else:
            texto_valor = f"{valor:.2f}"
            largura = max(
                0,
                min((valor / maximo) * 100, 100)
            )

        subtitulo = ""

        if subtitulo_col is not None and subtitulo_col in linha.index:
            subtitulo = str(linha[subtitulo_col])

        render_html(
            f"""
            <div class="ranking-card">
                <div class="ranking-position">{indice}</div>

                <div class="ranking-content">
                    <div class="ranking-main">
                        <div>
                            <div class="ranking-title">
                                {html.escape(titulo)}
                            </div>

                            <div class="ranking-subtitle">
                                {html.escape(subtitulo)}
                            </div>
                        </div>

                        <div class="ranking-value">
                            {texto_valor}
                        </div>
                    </div>

                    <div class="mini-bar">
                        <div
                            class="bar-fill"
                            style="width:{largura:.2f}%">
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def mostrar_sem_dados():
    render_html(
        """
        <div class="empty-state">
            Não foi possível realizar esta análise porque as colunas necessárias
            não foram encontradas nos dados disponíveis.
        </div>
        """,
        unsafe_allow_html=True
    )


def resumo_insight(titulo, texto, tipo="neutral"):
    render_html(
        f"""
        <div class="insight-card insight-{tipo}">
            <div class="insight-title">{html.escape(titulo)}</div>
            <div class="insight-text">{html.escape(texto)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def obter_recall_col(df):
    return obter_coluna(
        df,
        [
            "item_recall_rate",
            "avg_session_recall",
            "avg_recall",
            "recall"
        ]
    )


carregar_css()

dados = carregar_dados()

traces = dados["traces"]
courses = dados["courses"]
curve = dados["curve"]
words = dados["words"]

idiomas = obter_idiomas(
    traces,
    courses,
    curve,
    words
)


with st.sidebar:
    render_html(
        """
        <div class="sidebar-brand">
            <div class="sidebar-title">Duolingo Learning Insights</div>
            <div class="sidebar-subtitle">Sistema de Apoio à Decisão</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    pagina = st.radio(
        "Navegação",
        [
            "Visão Geral",
            "Consulta de Pesquisa",
            "Análise por Idioma",
            "Esquecimento",
            "Revisões",
            "Dificuldades",
            "Simulador de Prioridade",
            "Decisões"
        ]
    )

    st.divider()

    st.caption("Parâmetros globais")

    meta_recall = (
        st.slider(
            "Meta de recall",
            min_value=50,
            max_value=100,
            value=85,
            step=1
        ) / 100
    )

    limite_critico = (
        st.slider(
            "Recall máximo para conteúdo crítico",
            min_value=20,
            max_value=90,
            value=65,
            step=1
        ) / 100
    )

    quantidade_prioridades = st.slider(
        "Quantidade de prioridades exibidas",
        min_value=5,
        max_value=30,
        value=10
    )

    st.divider()

    st.caption("Equipe")
    st.write("Ana Leticia")
    st.write("Denise Matos")
    st.write("Lana Liz")


if pagina == "Visão Geral":

    cabecalho(
        "Duolingo Learning Insights",
        "Sistema de Apoio à Decisão para investigar retenção, esquecimento, revisões e dificuldades no aprendizado de idiomas."
    )

    usuarios_col = obter_coluna(courses, ["n_users"])

    if usuarios_col is not None:
        total_usuarios = courses[usuarios_col].sum()
    else:
        user_col = obter_coluna(
            traces,
            ["user_id", "user"]
        )

        total_usuarios = (
            traces[user_col].nunique()
            if user_col is not None
            else 0
        )

    interacoes_col = obter_coluna(courses, ["n_traces"])

    if interacoes_col is not None:
        total_interacoes = courses[interacoes_col].sum()
    else:
        total_interacoes = len(traces)

    recall_col_courses = obter_recall_col(courses)
    recall_col_traces = obter_recall_col(traces)

    if recall_col_courses is not None:
        recall_medio = courses[recall_col_courses].mean()
    elif recall_col_traces is not None:
        recall_medio = traces[recall_col_traces].mean()
    else:
        recall_medio = np.nan

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        caixa_metrica(
            "Total de usuários",
            numero(total_usuarios),
            "Usuários disponíveis para análise"
        )

    with c2:
        caixa_metrica(
            "Total de interações",
            numero(total_interacoes),
            "Registros de aprendizagem"
        )

    with c3:
        caixa_metrica(
            "Recall médio",
            percentual(recall_medio),
            "Retenção média observada"
        )

    with c4:
        caixa_metrica(
            "Idiomas disponíveis",
            str(len(idiomas)),
            "Idiomas identificados na base"
        )

    secao(
        "Mapa de investigação",
        "As análises foram organizadas para responder perguntas sobre desempenho, esquecimento, revisões e dificuldade."
    )

    col1, col2 = st.columns(2)

    with col1:
        resumo_insight(
            "Retenção e esquecimento",
            "Avalie como o tempo desde a última prática está associado à redução do recall.",
            "purple"
        )

        resumo_insight(
            "Revisões",
            "Compare diferentes quantidades de exposições anteriores e observe o comportamento da retenção.",
            "blue"
        )

    with col2:
        resumo_insight(
            "Dificuldades",
            "Identifique palavras, conteúdos e classes gramaticais associados aos maiores índices de erro.",
            "orange"
        )

        resumo_insight(
            "Decisão",
            "Utilize o Índice de Prioridade de Revisão para transformar os resultados analíticos em recomendações.",
            "red"
        )

    secao(
        "Leitura rápida dos dados",
        "Selecione um idioma para observar sua posição em relação aos demais."
    )

    idioma_comparacao = st.selectbox(
        "Idioma para comparação",
        idiomas if idiomas else ["Não disponível"],
        key="visao_geral_idioma"
    )

    df_idioma = filtrar_idioma(courses, idioma_comparacao)
    recall_col = obter_recall_col(df_idioma)

    if recall_col is not None and not df_idioma.empty:
        recall_idioma = df_idioma[recall_col].mean()
        diferenca = recall_idioma - recall_medio

        c1, c2, c3 = st.columns(3)

        with c1:
            caixa_metrica(
                "Recall do idioma",
                percentual(recall_idioma),
                "Média observada"
            )

        with c2:
            caixa_metrica(
                "Meta definida",
                percentual(meta_recall),
                "Parâmetro global"
            )

        with c3:
            caixa_metrica(
                "Diferença para a média",
                f"{diferenca * 100:+.1f} p.p.",
                "Comparação com todos os idiomas"
            )

        if recall_idioma >= meta_recall:
            resumo_insight(
                "Situação",
                "O idioma selecionado atingiu a meta global de recall definida no painel.",
                "green"
            )
        else:
            resumo_insight(
                "Situação",
                "O idioma selecionado está abaixo da meta global e pode exigir maior atenção nas revisões.",
                "orange"
            )


elif pagina == "Consulta de Pesquisa":

    cabecalho(
        "Consulta de Pesquisa",
        "Escolha uma pergunta. O resultado é recalculado conforme o idioma e os critérios selecionados."
    )

    pergunta = st.selectbox(
        "Pergunta de pesquisa",
        [
            "Como o tempo sem prática afeta a retenção?",
            "Mais revisões melhoram o recall?",
            "Quais palavras são mais difíceis de aprender?",
            "Quais classes gramaticais geram mais erros?",
            "Quais cursos apresentam melhor desempenho?",
            "A probabilidade prevista corresponde aos acertos reais?"
        ]
    )

    idioma = st.selectbox(
        "Idioma",
        ["Todos"] + idiomas
    )

    if pergunta == "Como o tempo sem prática afeta a retenção?":

        df = filtrar_idioma(curve, idioma)

        lag_col = obter_coluna(
            df,
            ["lag_days", "avg_lag_days"]
        )

        recall_col = obter_recall_col(df)

        if lag_col is not None and recall_col is not None:
            resultado = (
                df.groupby(lag_col)[recall_col]
                .mean()
                .reset_index()
                .sort_values(lag_col)
            )

            resultado = resultado.dropna()

            if not resultado.empty:
                primeiro = resultado.iloc[0]
                ultimo = resultado.iloc[-1]

                variacao = (
                    ultimo[recall_col] - primeiro[recall_col]
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    caixa_metrica(
                        "Menor intervalo",
                        f"{float(primeiro[lag_col]):.1f} dias",
                        percentual(primeiro[recall_col])
                    )

                with c2:
                    caixa_metrica(
                        "Maior intervalo",
                        f"{float(ultimo[lag_col]):.1f} dias",
                        percentual(ultimo[recall_col])
                    )

                with c3:
                    caixa_metrica(
                        "Variação observada",
                        f"{variacao * 100:+.1f} p.p.",
                        "Diferença entre os extremos"
                    )

                secao(
                    "Faixas de retenção",
                    "Cada linha representa a retenção média associada ao tempo sem prática."
                )

                ranking_cards(
                    resultado,
                    lag_col,
                    recall_col,
                    max_linhas=quantidade_prioridades,
                    percentual_valor=True
                )

                if variacao < 0:
                    resumo_insight(
                        "Resposta da análise",
                        "Os dados selecionados indicam redução do recall à medida que o intervalo sem prática aumenta.",
                        "orange"
                    )
                else:
                    resumo_insight(
                        "Resposta da análise",
                        "Os dados selecionados não indicam uma redução linear do recall entre os intervalos observados.",
                        "blue"
                    )
            else:
                mostrar_sem_dados()
        else:
            mostrar_sem_dados()

    elif pergunta == "Mais revisões melhoram o recall?":

        df = filtrar_idioma(curve, idioma)

        exposure_col = obter_coluna(
            df,
            [
                "avg_prior_exposures",
                "prior_exposures",
                "exposures"
            ]
        )

        recall_col = obter_recall_col(df)

        if exposure_col is not None and recall_col is not None:
            resultado = (
                df.groupby(exposure_col)[recall_col]
                .mean()
                .reset_index()
                .sort_values(exposure_col)
            )

            resultado = resultado.dropna()

            if not resultado.empty:
                primeira = resultado.iloc[0]
                ultima = resultado.iloc[-1]

                ganho = (
                    ultima[recall_col] - primeira[recall_col]
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    caixa_metrica(
                        "Menor número de exposições",
                        valor_numerico(primeira[exposure_col]),
                        percentual(primeira[recall_col])
                    )

                with c2:
                    caixa_metrica(
                        "Maior número de exposições",
                        valor_numerico(ultima[exposure_col]),
                        percentual(ultima[recall_col])
                    )

                with c3:
                    caixa_metrica(
                        "Variação do recall",
                        f"{ganho * 100:+.1f} p.p.",
                        "Entre os extremos observados"
                    )

                secao(
                    "Comparação por quantidade de revisões",
                    "Altere o idioma e observe como a relação entre exposições e recall muda."
                )

                ranking_cards(
                    resultado,
                    exposure_col,
                    recall_col,
                    max_linhas=quantidade_prioridades,
                    percentual_valor=True
                )

                if ganho > 0:
                    resumo_insight(
                        "Resposta da análise",
                        "Nos extremos analisados, um número maior de exposições está associado a maior recall médio.",
                        "green"
                    )
                else:
                    resumo_insight(
                        "Resposta da análise",
                        "Nos extremos analisados, o aumento das exposições não apresentou ganho médio de recall.",
                        "orange"
                    )
            else:
                mostrar_sem_dados()
        else:
            mostrar_sem_dados()

    elif pergunta == "Quais palavras são mais difíceis de aprender?":

        df = filtrar_idioma(words, idioma)

        palavra_col = obter_nome_palavra(df)
        recall_col = obter_recall_col(df)

        busca = st.text_input(
            "Pesquisar palavra ou parte da palavra"
        )

        if busca and palavra_col is not None:
            df = df[
                df[palavra_col]
                .astype(str)
                .str.contains(
                    busca,
                    case=False,
                    na=False
                )
            ]

        if palavra_col is not None and recall_col is not None:
            resultado = (
                df.sort_values(recall_col)
                [[palavra_col, recall_col] + (
                    ["classe_gramatical"]
                    if "classe_gramatical" in df.columns
                    else []
                )]
            )

            secao(
                "Ranking de dificuldade",
                "Palavras com menor recall aparecem primeiro."
            )

            ranking_cards(
                resultado,
                palavra_col,
                recall_col,
                max_linhas=quantidade_prioridades,
                percentual_valor=True,
                subtitulo_col=(
                    "classe_gramatical"
                    if "classe_gramatical" in resultado.columns
                    else None
                )
            )

            if not resultado.empty:
                palavra_dificil = resultado.iloc[0][palavra_col]
                recall_dificil = resultado.iloc[0][recall_col]

                resumo_insight(
                    "Conteúdo com maior dificuldade",
                    f"A palavra {palavra_dificil} apresentou recall médio de {recall_dificil * 100:.1f}% no recorte selecionado.",
                    "red"
                )
        else:
            mostrar_sem_dados()

    elif pergunta == "Quais classes gramaticais geram mais erros?":

        df = filtrar_idioma(words, idioma)
        recall_col = obter_recall_col(df)

        if (
            "classe_gramatical" in df.columns
            and recall_col is not None
        ):
            resultado = (
                df.groupby("classe_gramatical")
                .agg(
                    quantidade=("classe_gramatical", "size"),
                    recall_medio=(recall_col, "mean")
                )
                .reset_index()
            )

            resultado["taxa_erro"] = (
                1 - resultado["recall_medio"]
            )

            resultado = resultado.sort_values(
                "taxa_erro",
                ascending=False
            )

            secao(
                "Ranking por taxa de erro",
                "Quanto maior a barra, maior a proporção média estimada de erros."
            )

            ranking_cards(
                resultado,
                "classe_gramatical",
                "taxa_erro",
                max_linhas=quantidade_prioridades,
                percentual_valor=True,
                subtitulo_col="quantidade"
            )

            if not resultado.empty:
                classe = resultado.iloc[0]["classe_gramatical"]
                erro = resultado.iloc[0]["taxa_erro"]

                resumo_insight(
                    "Classe com maior dificuldade",
                    f"{classe} apresentou a maior taxa média de erro no recorte atual: {erro * 100:.1f}%.",
                    "red"
                )
        else:
            mostrar_sem_dados()

    elif pergunta == "Quais cursos apresentam melhor desempenho?":

        df = filtrar_idioma(courses, idioma)

        course_col = obter_coluna(
            df,
            ["course", "course_name"]
        )

        recall_col = obter_recall_col(df)

        if course_col is not None and recall_col is not None:
            resultado = (
                df.sort_values(
                    recall_col,
                    ascending=False
                )
                [[course_col, recall_col]]
            )

            secao(
                "Ranking dos cursos",
                "Os cursos são organizados pelo maior recall médio."
            )

            ranking_cards(
                resultado,
                course_col,
                recall_col,
                max_linhas=quantidade_prioridades,
                percentual_valor=True
            )

            if not resultado.empty:
                melhor = resultado.iloc[0][course_col]
                recall = resultado.iloc[0][recall_col]

                resumo_insight(
                    "Melhor desempenho",
                    f"{melhor} ocupa a primeira posição no recorte atual, com recall médio de {recall * 100:.1f}%.",
                    "green"
                )
        else:
            mostrar_sem_dados()

    elif pergunta == "A probabilidade prevista corresponde aos acertos reais?":

        prediction_col = obter_coluna(
            traces,
            ["p_recall", "predicted_recall"]
        )

        real_col = obter_coluna(
            traces,
            ["session_correct", "correct"]
        )

        if prediction_col is not None and real_col is not None:
            df = traces[
                [prediction_col, real_col]
            ].dropna().copy()

            if not df.empty:
                df["faixa_previsao"] = pd.cut(
                    df[prediction_col],
                    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.01],
                    include_lowest=True
                )

                resultado = (
                    df.groupby(
                        "faixa_previsao",
                        observed=True
                    )
                    .agg(
                        previsao_media=(
                            prediction_col,
                            "mean"
                        ),
                        acerto_real=(
                            real_col,
                            "mean"
                        ),
                        interacoes=(
                            real_col,
                            "size"
                        )
                    )
                    .reset_index()
                )

                resultado["diferenca_absoluta"] = abs(
                    resultado["previsao_media"]
                    - resultado["acerto_real"]
                )

                diferenca_media = (
                    resultado["diferenca_absoluta"].mean()
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    caixa_metrica(
                        "Previsão média",
                        percentual(df[prediction_col].mean()),
                        "Probabilidade estimada"
                    )

                with c2:
                    caixa_metrica(
                        "Acerto real médio",
                        percentual(df[real_col].mean()),
                        "Resultado observado"
                    )

                with c3:
                    caixa_metrica(
                        "Diferença média",
                        percentual(diferenca_media),
                        "Distância entre previsão e realidade"
                    )

                secao(
                    "Calibração por faixa",
                    "Compare diretamente a previsão média com a taxa real de acerto."
                )

                for _, linha in resultado.iterrows():
                    faixa = str(linha["faixa_previsao"])
                    previsto = linha["previsao_media"]
                    real = linha["acerto_real"]
                    diferenca = linha["diferenca_absoluta"]

                    render_html(
                        f"""
                        <div class="calibration-card">
                            <div class="calibration-header">
                                <div class="calibration-title">
                                    Faixa prevista: {html.escape(faixa)}
                                </div>
                                <div class="calibration-difference">
                                    Diferença: {diferenca * 100:.1f} p.p.
                                </div>
                            </div>

                            <div class="comparison-row">
                                <span>Previsto</span>
                                <div class="comparison-track">
                                    <div
                                        class="comparison-predicted"
                                        style="width:{previsto * 100:.2f}%">
                                    </div>
                                </div>
                                <strong>{previsto * 100:.1f}%</strong>
                            </div>

                            <div class="comparison-row">
                                <span>Real</span>
                                <div class="comparison-track">
                                    <div
                                        class="comparison-real"
                                        style="width:{real * 100:.2f}%">
                                    </div>
                                </div>
                                <strong>{real * 100:.1f}%</strong>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if diferenca_media <= 0.05:
                    resumo_insight(
                        "Diagnóstico",
                        "A diferença média entre as probabilidades previstas e os acertos reais é baixa, indicando boa aproximação no recorte analisado.",
                        "green"
                    )
                else:
                    resumo_insight(
                        "Diagnóstico",
                        "Existe uma diferença relevante entre previsão e resultado real em parte das faixas analisadas.",
                        "orange"
                    )
        else:
            mostrar_sem_dados()


elif pagina == "Análise por Idioma":

    cabecalho(
        "Análise por Idioma",
        "Selecione um idioma e altere o tipo de análise para investigar diferentes aspectos do desempenho."
    )

    if not idiomas:
        st.warning("Nenhum idioma foi identificado nos arquivos.")
        st.stop()

    idioma = st.selectbox(
        "Idioma selecionado",
        idiomas
    )

    tipo_analise = st.radio(
        "Aspecto",
        [
            "Desempenho geral",
            "Palavras difíceis",
            "Classes gramaticais"
        ],
        horizontal=True
    )

    if tipo_analise == "Desempenho geral":

        df = filtrar_idioma(courses, idioma)

        recall_col = obter_recall_col(df)

        lag_col = obter_coluna(
            df,
            ["avg_lag_days", "lag_days"]
        )

        exposure_col = obter_coluna(
            df,
            [
                "avg_prior_exposures",
                "prior_exposures"
            ]
        )

        recall = (
            df[recall_col].mean()
            if recall_col is not None
            else np.nan
        )

        lag = (
            df[lag_col].mean()
            if lag_col is not None
            else np.nan
        )

        exposure = (
            df[exposure_col].mean()
            if exposure_col is not None
            else np.nan
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            caixa_metrica(
                "Recall médio",
                percentual(recall),
                "Retenção média do idioma"
            )

        with c2:
            caixa_metrica(
                "Intervalo médio",
                (
                    f"{lag:.1f} dias"
                    if not pd.isna(lag)
                    else "Não disponível"
                ),
                "Tempo entre práticas"
            )

        with c3:
            caixa_metrica(
                "Exposições médias",
                (
                    f"{exposure:.1f}"
                    if not pd.isna(exposure)
                    else "Não disponível"
                ),
                "Práticas anteriores"
            )

        if not pd.isna(recall):
            progresso = max(0, min(recall * 100, 100))
            diferenca_meta = (recall - meta_recall) * 100
            situacao_meta = (
                f"{diferenca_meta:+.1f} p.p. em relação à meta"
            )

            render_html(
                f"""
                <div class="performance-panel">
                    <div class="performance-header">
                        <div>
                            <div class="performance-label">
                                Desempenho de recall
                            </div>
                            <div class="performance-title">
                                {html.escape(str(idioma))}
                            </div>
                        </div>

                        <div class="performance-value">
                            {recall * 100:.1f}%
                        </div>
                    </div>

                    <div class="progress-container">
                        <div
                            class="progress-bar"
                            style="width:{progresso:.2f}%">
                        </div>
                    </div>

                    <div class="progress-meta">
                        Meta: {meta_recall * 100:.0f}% &nbsp;•&nbsp; {situacao_meta}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    elif tipo_analise == "Palavras difíceis":

        df = filtrar_idioma(words, idioma)

        recall_col = obter_recall_col(df)
        palavra_col = obter_nome_palavra(df)

        if recall_col is not None and palavra_col is not None:
            resultado = (
                df.sort_values(recall_col)
                [[palavra_col, recall_col] + (
                    ["classe_gramatical"]
                    if "classe_gramatical" in df.columns
                    else []
                )]
            )

            secao(
                f"Conteúdos mais difíceis em {idioma}",
                "Ranking baseado no menor recall médio."
            )

            ranking_cards(
                resultado,
                palavra_col,
                recall_col,
                max_linhas=quantidade_prioridades,
                percentual_valor=True,
                subtitulo_col=(
                    "classe_gramatical"
                    if "classe_gramatical" in resultado.columns
                    else None
                )
            )
        else:
            mostrar_sem_dados()

    else:

        df = filtrar_idioma(words, idioma)
        recall_col = obter_recall_col(df)

        if (
            "classe_gramatical" in df.columns
            and recall_col is not None
        ):
            resultado = (
                df.groupby("classe_gramatical")
                .agg(
                    recall_medio=(recall_col, "mean"),
                    quantidade=("classe_gramatical", "size")
                )
                .reset_index()
            )

            resultado["taxa_erro"] = (
                1 - resultado["recall_medio"]
            )

            resultado = resultado.sort_values(
                "taxa_erro",
                ascending=False
            )

            ranking_cards(
                resultado,
                "classe_gramatical",
                "taxa_erro",
                max_linhas=quantidade_prioridades,
                percentual_valor=True,
                subtitulo_col="quantidade"
            )
        else:
            mostrar_sem_dados()


elif pagina == "Esquecimento":

    cabecalho(
        "Esquecimento",
        "Explore os intervalos sem prática e identifique em que momentos a retenção passa a exigir maior atenção."
    )

    idioma = st.selectbox(
        "Idioma",
        ["Todos"] + idiomas
    )

    df = filtrar_idioma(curve, idioma)

    lag_col = obter_coluna(
        df,
        ["lag_days", "avg_lag_days"]
    )

    recall_col = obter_recall_col(df)

    if lag_col is not None and recall_col is not None:

        valores = pd.to_numeric(
            df[lag_col],
            errors="coerce"
        )

        if valores.notna().any():
            minimo = float(valores.min())
            maximo = float(valores.max())

            intervalo = st.slider(
                "Intervalo analisado sem prática",
                min_value=minimo,
                max_value=maximo,
                value=(minimo, maximo)
            )

            df = df[
                (valores >= intervalo[0])
                & (valores <= intervalo[1])
            ]

        resultado = (
            df.groupby(lag_col)[recall_col]
            .mean()
            .reset_index()
            .sort_values(lag_col)
        )

        if not resultado.empty:
            resultado["recall_percentual"] = (
                resultado[recall_col] * 100
            )

            primeiro = resultado.iloc[0]
            ultimo = resultado.iloc[-1]

            c1, c2, c3 = st.columns(3)

            with c1:
                caixa_metrica(
                    "Primeiro intervalo",
                    f"{primeiro[lag_col]:.1f} dias",
                    percentual(primeiro[recall_col])
                )

            with c2:
                caixa_metrica(
                    "Último intervalo",
                    f"{ultimo[lag_col]:.1f} dias",
                    percentual(ultimo[recall_col])
                )

            with c3:
                caixa_metrica(
                    "Queda acumulada",
                    f"{(primeiro[recall_col] - ultimo[recall_col]) * 100:.1f} p.p.",
                    "Entre o primeiro e último intervalo"
                )

            secao(
                "Escala de retenção",
                "As barras representam o recall médio observado em cada intervalo."
            )

            ranking_cards(
                resultado,
                lag_col,
                recall_col,
                max_linhas=quantidade_prioridades,
                percentual_valor=True
            )

            abaixo_meta = resultado[
                resultado[recall_col] < meta_recall
            ]

            if not abaixo_meta.empty:
                primeiro_risco = abaixo_meta.iloc[0][lag_col]

                resumo_insight(
                    "Momento de atenção",
                    f"No recorte selecionado, o recall fica abaixo da meta definida a partir de aproximadamente {primeiro_risco:.1f} dias.",
                    "orange"
                )
            else:
                resumo_insight(
                    "Momento de atenção",
                    "Nenhum dos intervalos exibidos ficou abaixo da meta global de recall.",
                    "green"
                )
        else:
            mostrar_sem_dados()
    else:
        mostrar_sem_dados()


elif pagina == "Revisões":

    cabecalho(
        "Revisões",
        "Compare diferentes quantidades de exposições anteriores e observe como o recall varia."
    )

    idioma = st.selectbox(
        "Idioma",
        ["Todos"] + idiomas
    )

    df = filtrar_idioma(curve, idioma)

    exposure_col = obter_coluna(
        df,
        [
            "avg_prior_exposures",
            "prior_exposures",
            "exposures"
        ]
    )

    recall_col = obter_recall_col(df)

    if exposure_col is not None and recall_col is not None:

        resultado = (
            df.groupby(exposure_col)[recall_col]
            .mean()
            .reset_index()
            .sort_values(exposure_col)
        )

        if not resultado.empty:
            opcoes = resultado[
                exposure_col
            ].astype(str).tolist()

            selecionadas = st.multiselect(
                "Exposições para comparação",
                opcoes,
                default=opcoes[:min(8, len(opcoes))]
            )

            if selecionadas:
                resultado = resultado[
                    resultado[exposure_col]
                    .astype(str)
                    .isin(selecionadas)
                ]

            resultado = resultado.copy()

            secao(
                "Comparador de exposições",
                "Selecione ou remova níveis de exposição para alterar a comparação."
            )

            ranking_cards(
                resultado,
                exposure_col,
                recall_col,
                max_linhas=len(resultado),
                percentual_valor=True
            )

            if len(resultado) >= 2:
                primeiro = resultado.iloc[0]
                ultimo = resultado.iloc[-1]

                ganho = (
                    ultimo[recall_col]
                    - primeiro[recall_col]
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    caixa_metrica(
                        "Menor exposição",
                        valor_numerico(primeiro[exposure_col]),
                        percentual(primeiro[recall_col])
                    )

                with c2:
                    caixa_metrica(
                        "Maior exposição",
                        valor_numerico(ultimo[exposure_col]),
                        percentual(ultimo[recall_col])
                    )

                with c3:
                    caixa_metrica(
                        "Variação",
                        f"{ganho * 100:+.1f} p.p.",
                        "Entre os extremos selecionados"
                    )

                if ganho > 0:
                    resumo_insight(
                        "Decisão sugerida",
                        "O cenário selecionado apresenta ganho de recall entre os níveis mínimo e máximo de exposição.",
                        "green"
                    )
                else:
                    resumo_insight(
                        "Decisão sugerida",
                        "O aumento das exposições não apresentou ganho no recorte selecionado.",
                        "orange"
                    )
        else:
            mostrar_sem_dados()
    else:
        mostrar_sem_dados()


elif pagina == "Dificuldades":

    cabecalho(
        "Dificuldades de Aprendizagem",
        "Pesquise conteúdos e combine filtros para encontrar palavras com maior risco de esquecimento."
    )

    idioma = st.selectbox(
        "Idioma",
        ["Todos"] + idiomas
    )

    df = filtrar_idioma(words, idioma)

    recall_col = obter_recall_col(df)
    palavra_col = obter_nome_palavra(df)

    col1, col2 = st.columns(2)

    with col1:
        busca = st.text_input(
            "Pesquisar palavra"
        )

    with col2:
        if "classe_gramatical" in df.columns:
            classes = sorted(
                df["classe_gramatical"]
                .dropna()
                .astype(str)
                .unique()
            )
        else:
            classes = []

        classe = st.selectbox(
            "Classe gramatical",
            ["Todas"] + classes
        )

    if busca and palavra_col is not None:
        df = df[
            df[palavra_col]
            .astype(str)
            .str.contains(
                busca,
                case=False,
                na=False
            )
        ]

    if (
        classe != "Todas"
        and "classe_gramatical" in df.columns
    ):
        df = df[
            df["classe_gramatical"] == classe
        ]

    if recall_col is not None and palavra_col is not None:

        df = df.copy()

        criticos = df[
            df[recall_col] <= limite_critico
        ].sort_values(recall_col)

        c1, c2, c3 = st.columns(3)

        with c1:
            caixa_metrica(
                "Conteúdos críticos",
                str(len(criticos)),
                f"Recall até {limite_critico * 100:.0f}%"
            )

        with c2:
            caixa_metrica(
                "Meta de recall",
                percentual(meta_recall),
                "Objetivo definido"
            )

        with c3:
            pior_recall = (
                criticos[recall_col].min()
                if not criticos.empty
                else np.nan
            )

            caixa_metrica(
                "Menor recall encontrado",
                percentual(pior_recall),
                "No filtro atual"
            )

        secao(
            "Conteúdos prioritários",
            "O ranking muda conforme a palavra pesquisada, idioma e classe gramatical."
        )

        ranking_cards(
            criticos,
            palavra_col,
            recall_col,
            max_linhas=quantidade_prioridades,
            percentual_valor=True,
            subtitulo_col=(
                "classe_gramatical"
                if "classe_gramatical" in criticos.columns
                else None
            )
        )

        if not criticos.empty:
            palavra = criticos.iloc[0][palavra_col]
            recall = criticos.iloc[0][recall_col]

            resumo_insight(
                "Maior prioridade no filtro atual",
                f"{palavra} apresenta recall de {recall * 100:.1f}% e merece atenção nas próximas revisões.",
                "red"
            )
    else:
        mostrar_sem_dados()


elif pagina == "Simulador de Prioridade":

    cabecalho(
        "Simulador de Prioridade de Revisão",
        "Altere o cenário e veja como o risco de esquecimento e a prioridade de revisão são recalculados."
    )

    secao(
        "Configuração do cenário",
        "Os parâmetros abaixo podem ser modificados livremente durante a apresentação."
    )

    col1, col2 = st.columns(2)

    with col1:
        idioma = st.selectbox(
            "Idioma",
            ["Todos"] + idiomas,
            key="simulador_idioma"
        )

        dias_sem_pratica = st.number_input(
            "Tempo desde a última prática (dias)",
            min_value=0,
            max_value=365,
            value=14,
            step=1
        )

    with col2:
        exposicoes = st.number_input(
            "Exposições anteriores",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )

        recall_minimo = (
            st.slider(
                "Recall mínimo desejado",
                min_value=50,
                max_value=100,
                value=85,
                step=1,
                key="simulador_recall"
            ) / 100
        )

    modo_dificuldade = st.radio(
        "Dificuldade do conteúdo",
        [
            "Utilizar dificuldade média do idioma",
            "Selecionar uma palavra específica"
        ],
        horizontal=True
    )

    palavra_selecionada = "Automática"

    if modo_dificuldade == "Selecionar uma palavra específica":

        df_palavras = filtrar_idioma(
            words,
            idioma
        )

        palavra_col = obter_nome_palavra(df_palavras)

        if palavra_col is not None:
            lista_palavras = sorted(
                df_palavras[palavra_col]
                .dropna()
                .astype(str)
                .unique()
            )

            if lista_palavras:
                palavra_selecionada = st.selectbox(
                    "Palavra",
                    lista_palavras
                )

    analisar = st.button(
        "Analisar prioridade"
    )

    if analisar:

        recall_estimado = estimar_recall_cenario(
            curve=curve,
            idioma=idioma,
            dias_sem_pratica=dias_sem_pratica,
            exposicoes=exposicoes,
            recall_historico=recall_minimo
        )

        dificuldade, origem_dificuldade = (
            calcular_dificuldade_palavra(
                words=words,
                idioma=idioma,
                palavra=palavra_selecionada
            )
        )

        indice = calcular_indice_prioridade(
            recall=recall_estimado,
            dias_sem_pratica=dias_sem_pratica,
            exposicoes=exposicoes,
            dificuldade=dificuldade
        )

        prioridade = classificar_prioridade(indice)

        recomendacao = obter_recomendacao(prioridade)

        risco_esquecimento = max(
            0,
            min(round(indice, 1), 100)
        )

        render_html(
            f"""
            <div class="simulator-result">
                <div class="simulator-risk-label">
                    RISCO DE ESQUECIMENTO
                </div>

                <div class="simulator-risk-value">
                    {risco_esquecimento:.1f}%
                </div>

                <div class="progress-container large-progress">
                    <div
                        class="progress-bar"
                        style="width:{risco_esquecimento}%">
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            caixa_metrica(
                "Índice de prioridade",
                f"{indice:.1f}",
                "Escala de 0 a 100"
            )

        with c2:
            caixa_metrica(
                "Prioridade",
                prioridade,
                "Classificação do cenário"
            )

        with c3:
            caixa_metrica(
                "Recall estimado",
                percentual(recall_estimado),
                "Resultado projetado"
            )

        render_html(
            f"""
            <div class="priority-result">
                <div class="priority-result-label">
                    Prioridade
                </div>

                <div class="priority-result-value">
                    {html.escape(prioridade.upper())}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        render_html(
            f"""
            <div class="recommendation">
                <div class="recommendation-title">
                    Recomendação
                </div>

                <div>
                    {html.escape(recomendacao)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        secao(
            "Como o resultado foi calculado",
            "O índice é uma regra analítica criada pelo projeto."
        )

        tabela_classificacao = pd.DataFrame(
            {
                "Índice": [
                    "0–30",
                    "31–60",
                    "61–80",
                    "81–100"
                ],
                "Prioridade": [
                    "Baixa",
                    "Média",
                    "Alta",
                    "Crítica"
                ]
            }
        )

        tabela_html(
            tabela_classificacao,
            max_linhas=4
        )

        resumo_insight(
            "Cenário utilizado",
            (
                f"Recall estimado: {recall_estimado * 100:.1f}% | "
                f"Tempo sem prática: {dias_sem_pratica} dias | "
                f"Exposições: {exposicoes} | "
                f"Dificuldade: {dificuldade * 100:.1f}%"
            ),
            "purple"
        )

        render_html(
            """
            <div class="methodology">
                <div class="methodology-title">
                    Índice de Prioridade de Revisão
                </div>

                <div>
                    O indicador combina baixo recall, tempo desde a última prática,
                    quantidade de exposições anteriores e dificuldade do conteúdo.
                    A métrica foi criada pelo projeto para apoiar a decisão sobre
                    quais conteúdos devem ser revisados primeiro.
                </div>

                <div class="methodology-weights">
                    Recall: 40% |
                    Tempo: 30% |
                    Exposições: 15% |
                    Dificuldade: 15%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


elif pagina == "Decisões":

    cabecalho(
        "Painel de Decisão",
        "Os conteúdos são classificados pelo Índice de Prioridade de Revisão para apoiar a escolha do que revisar primeiro."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        tempo_padrao = st.slider(
            "Tempo de referência sem prática",
            min_value=1,
            max_value=60,
            value=14
        )

    with col2:
        exposicoes_padrao = st.slider(
            "Exposições de referência",
            min_value=0,
            max_value=20,
            value=5
        )

    with col3:
        idioma = st.selectbox(
            "Idioma para priorização",
            ["Todos"] + idiomas
        )

    df_words = filtrar_idioma(
        words,
        idioma
    )

    prioridades = calcular_prioridades_conteudos(
        df_words,
        tempo_padrao=tempo_padrao,
        exposicoes_padrao=exposicoes_padrao
    )

    if prioridades.empty:

        st.warning(
            "Não foi possível calcular as prioridades com as colunas disponíveis."
        )

    else:

        baixa = len(
            prioridades[
                prioridades["prioridade"] == "Baixa"
            ]
        )

        media = len(
            prioridades[
                prioridades["prioridade"] == "Média"
            ]
        )

        alta = len(
            prioridades[
                prioridades["prioridade"] == "Alta"
            ]
        )

        critica = len(
            prioridades[
                prioridades["prioridade"] == "Crítica"
            ]
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            caixa_metrica(
                "Baixa",
                str(baixa),
                "Sem urgência"
            )

        with c2:
            caixa_metrica(
                "Média",
                str(media),
                "Programar revisão"
            )

        with c3:
            caixa_metrica(
                "Alta",
                str(alta),
                "Priorizar revisão"
            )

        with c4:
            caixa_metrica(
                "Crítica",
                str(critica),
                "Ação imediata"
            )

        secao(
            "Distribuição das prioridades",
            "A visualização mostra como os conteúdos foram classificados."
        )

        total = max(len(prioridades), 1)

        distribuicao = [
            ("Baixa", baixa, "priority-low-fill"),
            ("Média", media, "priority-medium-fill"),
            ("Alta", alta, "priority-high-fill"),
            ("Crítica", critica, "priority-critical-fill")
        ]

        for nome, quantidade, classe in distribuicao:
            percentual_distribuicao = (
                quantidade / total * 100
            )

            render_html(
                f"""
                <div class="distribution-card">
                    <div class="distribution-header">
                        <span>{nome}</span>
                        <strong>
                            {quantidade} conteúdos
                            ({percentual_distribuicao:.1f}%)
                        </strong>
                    </div>

                    <div class="distribution-track">
                        <div
                            class="{classe}"
                            style="width:{percentual_distribuicao:.2f}%">
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        secao(
            "Conteúdos prioritários",
            "O ranking abaixo é recalculado quando os parâmetros de tempo, exposições ou idioma são alterados."
        )

        palavra_col = obter_nome_palavra(prioridades)

        if palavra_col is not None:
            ranking = prioridades.sort_values(
                "indice_prioridade",
                ascending=False
            ).head(quantidade_prioridades)

            for indice, (_, linha) in enumerate(
                ranking.iterrows(),
                start=1
            ):
                palavra = linha[palavra_col]
                prioridade = linha["prioridade"]
                indice_prioridade = linha["indice_prioridade"]
                recall = linha["recall_utilizado"]

                subtitulo = []

                if "idioma" in linha.index:
                    subtitulo.append(str(linha["idioma"]))

                if "classe_gramatical" in linha.index:
                    subtitulo.append(
                        str(linha["classe_gramatical"])
                    )

                subtitulo = " · ".join(subtitulo)

                largura = max(
                    0,
                    min(float(indice_prioridade), 100)
                )

                render_html(
                    f"""
                    <div class="decision-ranking-card">
                        <div class="ranking-position">
                            {indice}
                        </div>

                        <div class="ranking-content">
                            <div class="decision-ranking-header">
                                <div>
                                    <div class="ranking-title">
                                        {html.escape(str(palavra))}
                                    </div>

                                    <div class="ranking-subtitle">
                                        {html.escape(subtitulo)}
                                    </div>
                                </div>

                                <div>
                                    <span class="status {classe_status(prioridade)}">
                                        {html.escape(prioridade)}
                                    </span>
                                </div>
                            </div>

                            <div class="decision-metrics">
                                <span>Recall: {recall * 100:.1f}%</span>
                                <span>Índice: {indice_prioridade:.1f}</span>
                            </div>

                            <div class="mini-bar">
                                <div
                                    class="bar-fill"
                                    style="width:{largura:.2f}%">
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if critica > 0:
            recomendacao = (
                "Existem conteúdos classificados como críticos. "
                "Eles devem ser priorizados nas próximas sessões de revisão."
            )
        elif alta > 0:
            recomendacao = (
                "Não há prioridade crítica, mas existem conteúdos com prioridade alta "
                "que devem receber atenção antes dos demais."
            )
        else:
            recomendacao = (
                "Os conteúdos analisados estão em níveis controlados. "
                "Mantenha o acompanhamento e a revisão periódica."
            )

        render_html(
            f"""
            <div class="recommendation">
                <div class="recommendation-title">
                    Decisão recomendada
                </div>

                <div>
                    {html.escape(recomendacao)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


render_html(
    """
    <div class="footer">
        Duolingo Learning Insights — Sistema de Apoio à Decisão<br>
        Ana Leticia · Denise Matos · Lana Liz
    </div>
    """,
    unsafe_allow_html=True
)