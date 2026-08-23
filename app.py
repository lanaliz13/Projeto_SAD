import os
import html
import textwrap
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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


def render_html(content):
    conteudo = textwrap.dedent(str(content)).strip()
    if hasattr(st, "html"):
        st.html(conteudo)
    else:
        st.markdown(conteudo, unsafe_allow_html=True)


def carregar_css():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(base_dir, "assets", "style.css")
    if os.path.exists(caminho):
        with open(caminho, "r", encoding="utf-8") as arquivo:
            css = arquivo.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def percentual(valor):
    return "Não disponível" if pd.isna(valor) or valor is None else f"{float(valor) * 100:.1f}%"


def numero(valor):
    return "0" if pd.isna(valor) or valor is None else f"{int(valor):,}".replace(",", ".")


def cabecalho(titulo, descricao):
    render_html(f"""
        <div class="app-header">
            <div class="app-eyebrow">SISTEMA DE APOIO À DECISÃO</div>
            <h1>{html.escape(titulo)}</h1>
            <p>{html.escape(descricao)}</p>
        </div>
    """)


def secao(titulo, descricao=""):
    render_html(f"""
        <div class="section-title">
            <h2>{html.escape(titulo)}</h2>
            <p>{html.escape(descricao)}</p>
        </div>
    """)


def caixa_metrica(titulo, valor, descricao):
    render_html(f"""
        <div class="metric-box">
            <div class="metric-label">{html.escape(str(titulo))}</div>
            <div class="metric-value">{html.escape(str(valor))}</div>
            <div class="metric-description">{html.escape(str(descricao))}</div>
        </div>
    """)


def filtrar_idioma(df, idioma):
    if df is None or df.empty or idioma == "Todos" or "idioma" not in df.columns:
        return df.copy() if df is not None else pd.DataFrame()
    resultado = df[df["idioma"].astype(str).str.lower() == str(idioma).lower()].copy()
    return resultado


def obter_idiomas(*dataframes):
    idiomas = set()
    for df in dataframes:
        if df is not None and not df.empty and "idioma" in df.columns:
            valores = df["idioma"].dropna().astype(str).unique()
            for v in valores:
                if v.lower() not in ['all', 'all languages', 'todos', 'nan']:
                    idiomas.add(v)
    return sorted(list(idiomas))


def obter_nome_palavra(df):
    return obter_coluna(df, ["surface_form", "lemma", "word"])


def obter_recall_col(df):
    return obter_coluna(df, ["item_recall_rate", "avg_session_recall", "avg_recall", "recall"])


carregar_css()
dados = carregar_dados()
traces, courses, curve, words = dados["traces"], dados["courses"], dados["curve"], dados["words"]
idiomas = obter_idiomas(traces, courses, curve, words)

with st.sidebar:
    render_html("""
        <div class="sidebar-brand">
            <div class="sidebar-title">Duolingo Insights</div>
            <div class="sidebar-subtitle">Painel Decision Support System</div>
        </div>
    """)
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
    st.caption("Parâmetros Globais")
    meta_recall = st.slider("Meta de recall", 50, 100, 85) / 100
    limite_critico = st.slider("Recall máx. p/ conteúdo crítico", 20, 90, 65) / 100
    quantidade_prioridades = st.slider("Qtd. de itens exibidos", 5, 50, 15)
    st.divider()
    st.caption("Equipe: Ana Leticia · Denise Matos · Lana Liz")

# --- VISÃO GERAL ---
if pagina == "Visão Geral":
    cabecalho("Duolingo Learning Insights", "Painel analítico para tomada de decisão no aprendizado de idiomas.")

    total_usuarios = courses["n_users"].sum() if not courses.empty and "n_users" in courses.columns else traces["user_id"].nunique() if not traces.empty and "user_id" in traces.columns else 0
    total_interacoes = courses["n_traces"].sum() if not courses.empty and "n_traces" in courses.columns else len(traces)
    recall_col = obter_recall_col(courses) or obter_recall_col(traces)
    recall_medio = courses[recall_col].mean() if recall_col and not courses.empty else np.nan

    c1, c2, c3, c4 = st.columns(4)
    with c1: caixa_metrica("Total de usuários", numero(total_usuarios), "Usuários na base")
    with c2: caixa_metrica("Total de interações", numero(total_interacoes), "Registros de treino")
    with c3: caixa_metrica("Recall médio", percentual(recall_medio), "Retenção geral")
    with c4: caixa_metrica("Idiomas", str(len(idiomas)), "Idiomas analisados")

    secao("Distribuição do Recall no Dataset")
    if not curve.empty:
        r_col = obter_recall_col(curve)
        if r_col:
            fig_overview = px.histogram(
                curve,
                x=r_col,
                nbins=25,
                title="Histograma de Densidade da Taxa de Recall",
                color_discrete_sequence=['#8b5cf6'],
                template="plotly_dark"
            )
            fig_overview.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Taxa de Recall",
                yaxis_title="Frequência"
            )
            st.plotly_chart(fig_overview, width="stretch")

# --- CONSULTA DE PESQUISA ---
elif pagina == "Consulta de Pesquisa":
    cabecalho("Consulta de Pesquisa", "Explore análises visuais baseadas nas perguntas centrais do sistema.")

    pergunta = st.selectbox("Selecione a Pergunta Analítica", [
        "Como o tempo sem prática afeta a retenção?",
        "Mais revisões melhoram o recall?",
        "Quais palavras são mais difíceis de aprender?",
    ])
    idioma = st.selectbox("Idioma", ["Todos"] + idiomas)

    if pergunta == "Como o tempo sem prática afeta a retenção?":
        df = filtrar_idioma(curve, idioma) if idioma != "Todos" else curve
        lag_col = obter_coluna(df, ["lag_bin", "lag_bin_order", "lag_days", "avg_lag_days"])
        recall_col = obter_recall_col(df)

        if not df.empty and lag_col and recall_col:

            ordem_lag = [
                "<1 hour",
                "1-6 hours",
                "6-24 hours",
                "1-3 days",
                "3-7 days",
                "1-2 weeks",
                "2-4 weeks",
                "1-3 months",
                "3+ months"
            ]

            df[lag_col] = pd.Categorical(
                df[lag_col],
                categories=ordem_lag,
                ordered=True
            )

            res = df.groupby(
                lag_col,
                as_index=False,
                observed=False
            )[recall_col].mean()

            fig = px.line(
                res,
                x=lag_col,
                y=recall_col,
                markers=True,
                title=f"Curva de Retenção pelo Tempo sem Prática — {idioma}",
                color_discrete_sequence=['#fb7185'],
                template="plotly_dark"
            )

            fig.add_hline(
                y=meta_recall,
                line_dash="dash",
                line_color="#34d399",
                annotation_text="Meta Global"
            )

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis_tickformat='.0%'
            )

            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Dados não encontrados para o filtro selecionado.")

    elif pergunta == "Mais revisões melhoram o recall?":
        df = filtrar_idioma(curve, idioma) if idioma != "Todos" else curve
        exp_col = obter_coluna(df, ["practice_bin", "avg_prior_exposures", "prior_exposures", "exposures"])
        recall_col = obter_recall_col(df)

        if not df.empty and exp_col and recall_col:

            ordem_exposicoes = [
                "1-2 exposures",
                "3-4 exposures",
                "5-9 exposures",
                "10-19 exposures",
                "20+ exposures"
            ]

            df[exp_col] = pd.Categorical(
                df[exp_col],
                categories=ordem_exposicoes,
                ordered=True
            )

            res = df.groupby(
                exp_col,
                as_index=False,
                observed=False
            )[recall_col].mean()

            fig = px.bar(
                res,
                x=exp_col,
                y=recall_col,
                title=f"Impacto das Exposições Anteriores no Recall — {idioma}",
                color_discrete_sequence=['#60a5fa'],
                template="plotly_dark"
            )

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis_tickformat='.0%'
            )

            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Dados não encontrados para o filtro selecionado.")

    elif pergunta == "Quais palavras são mais difíceis de aprender?":
        df = filtrar_idioma(words, idioma) if idioma != "Todos" else words
        palavra_col, recall_col = obter_nome_palavra(df), obter_recall_col(df)

        if not df.empty and palavra_col and recall_col:
            res = df.sort_values(recall_col).head(quantidade_prioridades)
            fig = px.bar(
                res, x=recall_col, y=palavra_col, orientation='h',
                title=f"Top {len(res)} Palavras Mais Difíceis — {idioma}",
                color=recall_col, color_continuous_scale='Reds_r', template="plotly_dark"
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_tickformat='.0%', yaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Dados não encontrados para o filtro selecionado.")


# --- ANÁLISE POR IDIOMA ---
elif pagina == "Análise por Idioma":
    cabecalho("Análise por Idioma", "Desempenho comparativo e detalhado do idioma selecionado.")
    idioma = st.selectbox("Idioma", idiomas if idiomas else ["German"])

    c_df = filtrar_idioma(courses, idioma)
    w_df = filtrar_idioma(words, idioma)

    r_col = obter_recall_col(c_df) or obter_recall_col(w_df)
    recall_id = c_df[r_col].mean() if not c_df.empty and r_col else w_df[r_col].mean() if not w_df.empty and r_col else np.nan

    c1, c2 = st.columns(2)
    with c1: caixa_metrica(f"Recall em {idioma}", percentual(recall_id), "Média calculada")
    with c2: caixa_metrica("Meta Estabelecida", percentual(meta_recall), "Parâmetro global")

    if not w_df.empty:
        p_col, r_col = obter_nome_palavra(w_df), obter_recall_col(w_df)
        if p_col and r_col:
            secao("Vocabulário Mais Crítico no Idioma")
            top_crit = w_df.sort_values(r_col).head(quantidade_prioridades)
            fig = px.bar(top_crit, x=r_col, y=p_col, orientation='h', color=r_col,
                         color_continuous_scale='Purples_r', template='plotly_dark',
                         title=f"Top {len(top_crit)} Palavras com Menor Retenção")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig, width="stretch")

# --- ESQUECIMENTO ---
elif pagina == "Esquecimento":
    cabecalho("Matriz de Esquecimento", "Cruzamento entre intervalos de prática e histórico de exposições.")
    idioma = st.selectbox("Idioma", ["Todos"] + idiomas)
    df = filtrar_idioma(curve, idioma) if idioma != "Todos" else curve

    lag_col = obter_coluna(df, ["lag_bin", "lag_days", "avg_lag_days"])
    exp_col = obter_coluna(df, ["practice_bin", "avg_prior_exposures", "prior_exposures", "exposures"])
    recall_col = obter_recall_col(df)

    if not df.empty and lag_col and exp_col and recall_col:

        ordem_lag = [
            "<1 hour",
            "1-6 hours",
            "6-24 hours",
            "1-3 days",
            "3-7 days",
            "1-2 weeks",
            "2-4 weeks",
            "1-3 months",
            "3+ months"
        ]

        ordem_exposicoes = [
            "1-2 exposures",
            "3-4 exposures",
            "5-9 exposures",
            "10-19 exposures",
            "20+ exposures"
        ]

        df[lag_col] = pd.Categorical(
            df[lag_col],
            categories=ordem_lag,
            ordered=True
        )

        df[exp_col] = pd.Categorical(
            df[exp_col],
            categories=ordem_exposicoes,
            ordered=True
        )

        pivot_df = df.pivot_table(
            index=exp_col,
            columns=lag_col,
            values=recall_col,
            aggfunc="mean",
            observed=False
        )

        fig = px.imshow(
            pivot_df,
            labels=dict(
                x="Intervalo sem Prática",
                y="Faixa de Exposições",
                color="Recall"
            ),
            x=pivot_df.columns,
            y=pivot_df.index,
            color_continuous_scale="Viridis",
            template="plotly_dark",
            aspect="auto"
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=f"Heatmap de Retenção — {idioma}"
        )

        st.plotly_chart(fig, width="stretch")

    else:
        st.info(
            "Registros insuficientes para formar a Matriz de Esquecimento neste filtro."
        )

# --- REVISÕES ---
elif pagina == "Revisões":
    cabecalho(
        "Análise Avançada de Revisões",
        "Comportamento da retenção sob diferentes frequências de repetição."
    )

    idioma = st.selectbox("Idioma", ["Todos"] + idiomas)
    df = filtrar_idioma(curve, idioma) if idioma != "Todos" else curve

    exp_col = obter_coluna(
        df,
        ["practice_bin", "avg_prior_exposures", "prior_exposures", "exposures"]
    )
    recall_col = obter_recall_col(df)

    if not df.empty and exp_col and recall_col:

        ordem_exposicoes = [
            "1-2 exposures",
            "3-4 exposures",
            "5-9 exposures",
            "10-19 exposures",
            "20+ exposures"
        ]

        df[exp_col] = pd.Categorical(
            df[exp_col],
            categories=ordem_exposicoes,
            ordered=True
        )

        res = (
            df.groupby(
                exp_col,
                as_index=False,
                observed=False
            )[recall_col]
            .agg(["mean", "std"])
            .reset_index()
            .dropna()
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=res[exp_col],
                y=res["mean"],
                error_y=dict(
                    type="data",
                    array=res["std"],
                    visible=True
                ),
                mode="lines+markers",
                name="Recall Médio ± Desvio",
                line=dict(color="#8b5cf6", width=3),
                marker=dict(size=8)
            )
        )

        fig.update_layout(
            title=f"Estabilidade da Retenção por Nível de Revisão — {idioma}",
            xaxis_title="Faixa de Prática / Exposições",
            yaxis_title="Recall Médio",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_tickformat=".0%"
        )

        st.plotly_chart(fig, width="stretch")

    else:
        st.info(
            "Registros insuficientes para a Análise de Revisões neste filtro."
        )

# --- DIFICULDADES ---
elif pagina == "Dificuldades":
    cabecalho("Mapeamento de Dificuldades", "Análise de Vulnerabilidade do Vocabulário (Taxa de Erro vs. Frequência).")
    idioma = st.selectbox("Idioma", ["Todos"] + idiomas)
    df = filtrar_idioma(words, idioma) if idioma != "Todos" else words

    palavra_col = obter_nome_palavra(df)
    recall_col = obter_recall_col(df)
    exp_col = obter_coluna(df, ["avg_prior_exposures", "n_traces", "prior_exposures", "exposures"])

    if not df.empty and palavra_col and recall_col:
        df["taxa_erro"] = 1 - df[recall_col]
        
        # Seleciona TOP N para não poluir o gráfico
        df_plot = df.sort_values("taxa_erro", ascending=False).head(quantidade_prioridades).copy()

        x_val = df_plot[exp_col] if exp_col else np.arange(len(df_plot))
        x_label = "Exposições Anteriores Médias" if exp_col else "Índice do Item"

        fig = px.scatter(
            df_plot,
            x=x_val,
            y="taxa_erro",
            hover_name=palavra_col,
            hover_data={"taxa_erro": ":.2%", "classe_gramatical": True if "classe_gramatical" in df_plot.columns else False},
            color="taxa_erro",
            color_continuous_scale="Reds",
            size="taxa_erro",
            size_max=18,
            title=f"Vulnerabilidade das TOP {len(df_plot)} Palavras — {idioma}",
            template="plotly_dark"
        )
        fig.add_hline(y=1 - limite_critico, line_dash="dot", line_color="#fb7185", annotation_text="Limite Crítico de Erro")
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_title="Taxa de Erro (1 - Recall)",
            xaxis_title=x_label,
            yaxis_tickformat='.0%'
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Dados de palavras indisponíveis para o idioma selecionado.")

# --- SIMULADOR DE PRIORIDADE ---
elif pagina == "Simulador de Prioridade":
    cabecalho("Simulador de Prioridade de Revisão", "Calcule em tempo real o risco e o índice de prioridade de revisão.")

    col1, col2 = st.columns(2)
    with col1:
        idioma = st.selectbox("Idioma", ["Todos"] + idiomas, key="sim_lang")
        dias_sem_pratica = st.number_input("Dias sem prática", 0, 365, 14)
    with col2:
        exposicoes = st.number_input("Exposições anteriores", 0, 100, 5)
        recall_minimo = st.slider("Recall mínimo desejado", 50, 100, 85) / 100

    if st.button("🔍 Calcular Prioridade de Revisão", width="stretch"):
        recall_est = estimar_recall_cenario(curve, idioma, dias_sem_pratica, exposicoes, recall_minimo)
        dif, orig = calcular_dificuldade_palavra(words, idioma)
        indice = calcular_indice_prioridade(recall_est, dias_sem_pratica, exposicoes, dif)
        prio = classificar_prioridade(indice)

        c1, c2, c3 = st.columns(3)
        with c1: caixa_metrica("Índice Calculado", f"{indice:.1f} / 100", "Escala do SAD")
        with c2: caixa_metrica("Prioridade", prio, "Classificação do algoritmo")
        with c3: caixa_metrica("Recall Estimado", percentual(recall_est), "Previsão de retenção")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=indice, title={'text': "Risco de Esquecimento (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#8b5cf6"},
                'steps': [
                    {'range': [0, 30], 'color': "#102b23"},
                    {'range': [30, 60], 'color': "#38270d"},
                    {'range': [60, 80], 'color': "#3b1720"},
                    {'range': [80, 100], 'color': "#4a1019"}
                ]
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, template="plotly_dark")
        st.plotly_chart(fig_gauge, width="stretch")

# --- DECISÕES ---
elif pagina == "Decisões":
    cabecalho("Painel Executivo de Decisão", "Classificação global dos conteúdos conforme o Índice de Prioridade de Revisão.")

    col1, col2, col3 = st.columns(3)
    with col1: tempo_padrao = st.slider("Tempo de referência sem prática (dias)", 1, 60, 14)
    with col2: exposicoes_padrao = st.slider("Exposições de referência", 0, 20, 5)
    with col3: idioma = st.selectbox("Idioma", ["Todos"] + idiomas)

    df_w = filtrar_idioma(words, idioma) if idioma != "Todos" else words
    prio_df = calcular_prioridades_conteudos(df_w, tempo_padrao, exposicoes_padrao)

    if not prio_df.empty:
        counts = prio_df["prioridade"].value_counts().reset_index()
        counts.columns = ["Prioridade", "Quantidade"]

        fig_donut = px.pie(
            counts, names="Prioridade", values="Quantidade",
            title=f"Distribuição Geral das Prioridades de Revisão — {idioma}",
            hole=0.5, color="Prioridade",
            color_discrete_map={"Baixa": "#34d399", "Média": "#f59e0b", "Alta": "#fb7185", "Crítica": "#ff4d67"},
            template="plotly_dark"
        )
        fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_donut, width="stretch")
    else:
        st.warning("Não foi possível calcular a distribuição de prioridades para este filtro.")

render_html("""
    <div class="footer">
        Duolingo Learning Insights — Sistema de Apoio à Decisão<br>
        Ana Leticia · Denise Matos · Lana Liz
    </div>
""")