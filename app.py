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
    calcular_prioridades_conteudos,
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


def explicacao_grafico(titulo, texto):
    render_html(f"""
        <div class="insight-card insight-purple" style="margin-bottom: 1.2rem;">
            <div class="insight-title">({html.escape(titulo)}):</div>
            <div class="insight-text">{html.escape(texto)}</div>
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
            "Decisões"
        ]
    )
    st.divider()
    st.caption("Parâmetros Globais")
    meta_recall = st.slider("Meta de recall", 50, 100, 85) / 100
    limite_critico = st.slider("Recall máx. p/ conteúdo crítico", 20, 90, 65) / 100
    quantidade_prioridades = st.slider("Qtd. de itens exibidos", 5, 30, 10)
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

    secao("Distribuição de Retenção na Base")
    explicacao_grafico(
        "Histograma de Recall",
        "Este gráfico mostra a quantidade de sessões divididas por sua taxa de acerto. Quanto mais barras concentradas à direita (próximas de 100%), melhor está a retenção geral dos alunos."
    )
    if not curve.empty:
        r_col = obter_recall_col(curve)
        if r_col:
            fig_overview = px.histogram(
                curve,
                x=r_col,
                nbins=20,
                title="Distribuição da Taxa de Recall (Retenção)",
                color_discrete_sequence=['#8b5cf6'],
                template="plotly_dark"
            )
            fig_overview.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Taxa de Retenção (0% a 100%)",
                yaxis_title="Frequência na Base",
                xaxis_tickformat='.0%'
            )
            st.plotly_chart(fig_overview, width="stretch")

# --- CONSULTA DE PESQUISA ---
elif pagina == "Consulta de Pesquisa":
    cabecalho("Consulta de Pesquisa", "Respostas visuais e diretas para as perguntas de negócio do sistema.")

    pergunta = st.selectbox("Selecione a Pergunta Analítica", [
        "Como o tempo sem prática afeta a retenção?",
        "Mais revisões melhoram o recall?",
        "Quais palavras são mais difíceis de aprender?",
        "Quais classes gramaticais geram mais erros?"
    ])
    idioma = st.selectbox("Idioma", ["Todos"] + idiomas)

    if pergunta == "Como o tempo sem prática afeta a retenção?":
        explicacao_grafico(
            "Tempo sem Prática vs. Retenção",
            "A linha mostra como o recall (acertos) cai à medida que os dias passam sem revisão. A linha tracejada verde é a meta do sistema (85%). Se a linha rosa cair abaixo da verde, o aluno precisa revisar o conteúdo."
        )
        df = filtrar_idioma(curve, idioma) if idioma != "Todos" else curve
        lag_col = obter_coluna(df, ["lag_bin", "lag_days", "avg_lag_days"])
        recall_col = obter_recall_col(df)

        if not df.empty and lag_col and recall_col:
            ordem_lag = ["<1 hour", "1-6 hours", "6-24 hours", "1-3 days", "3-7 days", "1-2 weeks", "2-4 weeks", "1-3 months", "3+ months"]
            df[lag_col] = pd.Categorical(df[lag_col], categories=ordem_lag, ordered=True)
            res = df.groupby(lag_col, as_index=False, observed=False)[recall_col].mean()

            fig = px.line(
                res, x=lag_col, y=recall_col, markers=True,
                title=f"Curva de Retenção pelo Tempo sem Prática — {idioma}",
                color_discrete_sequence=['#fb7185'], template="plotly_dark"
            )
            fig.add_hline(y=meta_recall, line_dash="dash", line_color="#34d399", annotation_text="Meta Global (85%)")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_tickformat='.0%', yaxis_title="Taxa de Acerto", xaxis_title="Intervalo Sem Prática")
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Dados não encontrados para o filtro selecionado.")

    elif pergunta == "Mais revisões melhoram o recall?":
        explicacao_grafico(
            "Quantidade de Revisões vs. Acertos",
            "Cada barra representa um nível de repetição. Barras mais altas indicam que quanto mais o aluno pratica aquela palavra, maior é a sua chance de lembrar dela corretamente nas sessões seguintes."
        )
        df = filtrar_idioma(curve, idioma) if idioma != "Todos" else curve
        exp_col = obter_coluna(df, ["practice_bin", "avg_prior_exposures", "prior_exposures"])
        recall_col = obter_recall_col(df)

        if not df.empty and exp_col and recall_col:
            ordem_exposicoes = ["1-2 exposures", "3-4 exposures", "5-9 exposures", "10-19 exposures", "20+ exposures"]
            df[exp_col] = pd.Categorical(df[exp_col], categories=ordem_exposicoes, ordered=True)
            res = df.groupby(exp_col, as_index=False, observed=False)[recall_col].mean()

            fig = px.bar(
                res, x=exp_col, y=recall_col, text_auto='.1%',
                title=f"Impacto das Exposições Anteriores no Recall — {idioma}",
                color_discrete_sequence=['#60a5fa'], template="plotly_dark"
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_tickformat='.0%', yaxis_title="Taxa de Acerto", xaxis_title="Número de Revisões Anteriores")
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Dados não encontrados para o filtro selecionado.")

    elif pergunta == "Quais palavras são mais difíceis de aprender?":
        explicacao_grafico(
            "Top Palavras com Menor Retenção",
            "As palavras no topo desta lista apresentam as menores taxas de acerto na base de dados. Elas representam o vocabulário mais complexo e que exige maior reforço pedagógico."
        )
        df = filtrar_idioma(words, idioma) if idioma != "Todos" else words
        palavra_col, recall_col = obter_nome_palavra(df), obter_recall_col(df)

        if not df.empty and palavra_col and recall_col:
            res = df.sort_values(recall_col).head(quantidade_prioridades)
            fig = px.bar(
                res, x=recall_col, y=palavra_col, orientation='h', text_auto='.1%',
                title=f"Top {len(res)} Palavras Mais Difíceis — {idioma}",
                color=recall_col, color_continuous_scale='Reds_r', template="plotly_dark"
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_tickformat='.0%', xaxis_title="Taxa de Acerto (Menor = Mais Difícil)", yaxis_title="Palavra", yaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Dados não encontrados para o filtro selecionado.")

    elif pergunta == "Quais classes gramaticais geram mais erros?":
        explicacao_grafico(
            "Distribuição de Erro por Gramática",
            "Este gráfico de pizza mostra a proporção de erros concentrada por categoria gramatical (ex: Verbos, Substantivos, Adjetivos). A fatia maior indica qual tipo de palavra mais confunde os alunos."
        )
        df = filtrar_idioma(words, idioma) if idioma != "Todos" else words
        recall_col = obter_recall_col(df)

        if not df.empty and "classe_gramatical" in df.columns and recall_col:
            res = df.groupby("classe_gramatical", as_index=False)[recall_col].mean()
            res["taxa_erro"] = 1 - res[recall_col]
            res = res.sort_values("taxa_erro", ascending=False)

            fig = px.pie(
                res, names="classe_gramatical", values="taxa_erro",
                title=f"Concentração de Erros por Categoria Gramatical — {idioma}",
                hole=0.4, template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
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
            fig = px.bar(top_crit, x=r_col, y=p_col, orientation='h', color=r_col, text_auto='.1%',
                         color_continuous_scale='Purples_r', template='plotly_dark',
                         title=f"Top {len(top_crit)} Palavras com Menor Retenção em {idioma}")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_tickformat='.0%', yaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig, width="stretch")

# --- ESQUECIMENTO ---
elif pagina == "Esquecimento":
    cabecalho(
        "Análise e Diagnóstico de Esquecimento",
        "Mapeamento da perda de retenção ao longo do tempo e identificação de conteúdos em estado crítico."
    )

    explicacao_grafico(
        "Diagnóstico da Curva do Esquecimento",
        "O heatmap apresenta a relação entre o tempo sem prática e o número de exposições anteriores. "
        "Células com menor recall representam maior risco de esquecimento e indicam conteúdos "
        "que podem necessitar de revisão."
    )

    c_esq1, c_esq2 = st.columns(2)

    with c_esq1:
        idioma = st.selectbox(
            "Idioma para análise",
            ["Todos"] + idiomas,
            key="esq_lang"
        )


    df = (
        filtrar_idioma(curve, idioma)
        if idioma != "Todos"
        else curve.copy()
    )

    lag_col = obter_coluna(
        df,
        [
            "lag_bin",
            "lag_days",
            "avg_lag_days"
        ]
    )

    exp_col = obter_coluna(
        df,
        [
            "practice_bin",
            "avg_prior_exposures",
            "prior_exposures"
        ]
    )

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

   
        pivot_df = pivot_df.reindex(
            index=ordem_exposicoes,
            columns=ordem_lag
        )

        valores_recall = pivot_df.values.flatten()
        valores_recall = valores_recall[
            ~pd.isna(valores_recall)
        ]


        secao(
            "Matriz de Esquecimento",
            "Relação entre tempo sem prática, exposições anteriores e retenção."
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
            aspect="auto",
            text_auto=".1%"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=f"Heatmap de Retenção — {idioma}",
            xaxis_title="Intervalo sem Prática",
            yaxis_title="Faixa de Exposições"
        )

        fig.update_coloraxes(
            colorbar_tickformat=".0%",
            colorbar_title="Recall"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        render_html(
            f"""
            <div class="insight-card insight-purple">
                <div class="insight-title">
                </div>

                <div class="insight-text">
                    As células representam a taxa média de recall para cada
                    combinação entre tempo sem prática e quantidade de
                    exposições anteriores.
                    <br><br>

                    <b>Recall mais alto</b> indica maior retenção.
                    <br>

                    <b>Recall mais baixo</b> indica maior risco de esquecimento.
                    <br><br>

                    O limite crítico configurado pelo sistema é de
                    <b>{limite_critico * 100:.0f}%</b>.
                    A meta global de retenção é de
                    <b>{meta_recall * 100:.0f}%</b>.
                </div>
            </div>
            """
        )


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

    explicacao_grafico(
        "Curva de Aprendizado por Prática",
        "A análise mostra como a retenção varia conforme aumenta "
        "o número de exposições anteriores ao conteúdo."
    )

    idioma = st.selectbox(
        "Idioma",
        ["Todos"] + idiomas,
        key="rev_lang"
    )

    df = (
        filtrar_idioma(curve, idioma)
        if idioma != "Todos"
        else curve.copy()
    )

    exp_col = obter_coluna(
        df,
        ["practice_bin", "avg_prior_exposures", "prior_exposures"]
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

        res = df.groupby(
            exp_col,
            as_index=False,
            observed=False
        )[recall_col].mean()

        res = res.sort_values(exp_col)

        fig = px.line(
            res,
            x=exp_col,
            y=recall_col,
            markers=True,
            text=res[recall_col].map(lambda x: f"{x:.1%}"),
            title=f"Evolução da Retenção por Histórico de Treino — {idioma}",
            template="plotly_dark"
        )

        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=10)
        )

        fig.update_layout(
            xaxis_title="Faixa de Prática / Exposições",
            yaxis_title="Recall Médio",
            yaxis_tickformat=".0%",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    else:
        st.info(
            "Registros insuficientes para a Análise de Revisões neste filtro."
        )

# --- DIFICULDADES ---
elif pagina == "Dificuldades":
    cabecalho("Mapeamento de Dificuldades", "Ranking de palavras que apresentam a maior taxa de erro dos alunos.")

    explicacao_grafico(
        "Mapeamento do Vocabulário Crítico",
        "Em vez de pontos amontoados, este gráfico lista de forma simples as palavras com maior taxa de erro (100% - taxa de acerto). A linha pontilhada indica o limite crítico tolerável de erro."
    )

    idioma = st.selectbox("Idioma", ["Todos"] + idiomas)
    df = filtrar_idioma(words, idioma) if idioma != "Todos" else words

    palavra_col = obter_nome_palavra(df)
    recall_col = obter_recall_col(df)

    if not df.empty and palavra_col and recall_col:
        df["taxa_erro"] = 1 - df[recall_col]
        df_plot = df.sort_values("taxa_erro", ascending=False).head(quantidade_prioridades).copy()

        fig = px.bar(
            df_plot,
            x="taxa_erro",
            y=palavra_col,
            orientation='h',
            text_auto='.1%',
            color="taxa_erro",
            color_continuous_scale="Reds",
            title=f"Top {len(df_plot)} Palavras com Maior Taxa de Erro — {idioma}",
            template="plotly_dark"
        )
        fig.add_vline(x=1 - limite_critico, line_dash="dot", line_color="#fb7185", annotation_text="Limite Crítico de Erro")
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Taxa de Erro Estimada",
            yaxis_title="Palavra",
            xaxis_tickformat='.0%',
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Dados de palavras indisponíveis para o idioma selecionado.")

# --- DECISÕES ---
elif pagina == "Decisões":

    cabecalho(
        "Decisões",
        "Priorize os conteúdos que mais precisam de revisão."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        tempo_padrao = st.slider(
            "Dias sem prática", 1, 60, 14, key="dec_tempo"
        )

    with col2:
        exposicoes_padrao = st.slider(
            "Exposições", 0, 20, 5, key="dec_exposicoes"
        )

    with col3:
        idioma = st.selectbox(
            "Idioma", ["Todos"] + idiomas, key="dec_idioma"
        )

    df_w = filtrar_idioma(words, idioma) if idioma != "Todos" else words

    prio_df = calcular_prioridades_conteudos(
        df_w,
        tempo_padrao,
        exposicoes_padrao
    )

    if prio_df.empty:
        st.info("Não foi possível calcular as prioridades para o filtro selecionado.")
    else:

        # Distribuição das prioridades
        counts = (
            prio_df["prioridade"]
            .value_counts()
            .reindex(
                ["Baixa", "Média", "Alta", "Crítica"],
                fill_value=0
            )
            .reset_index()
        )

        counts.columns = ["Prioridade", "Quantidade"]

        fig = px.pie(
            counts,
            names="Prioridade",
            values="Quantidade",
            title=f"Prioridade dos Conteúdos — {idioma}",
            hole=0.5,
            color="Prioridade",
            color_discrete_map={
                "Baixa": "#34d399",
                "Média": "#f59e0b",
                "Alta": "#fb7185",
                "Crítica": "#ff4d67"
            },
            template="plotly_dark"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(fig, width="stretch")

        # Ranking
        secao(
            "Conteúdos Prioritários",
            "Conteúdos com maior índice de prioridade."
        )

        palavra_col = obter_nome_palavra(prio_df)

        if palavra_col:

            ranking = (
                prio_df
                .sort_values("indice_prioridade", ascending=False)
                .head(quantidade_prioridades)
                .copy()
            )

            tabela = ranking[
                [
                    palavra_col,
                    "recall_utilizado",
                    "indice_prioridade",
                    "prioridade"
                ]
            ].copy()

            tabela["recall_utilizado"] = (
                tabela["recall_utilizado"] * 100
            ).round(1).astype(str) + "%"

            tabela["indice_prioridade"] = (
                tabela["indice_prioridade"].round(1)
            )

            tabela.columns = [
                "Palavra",
                "Recall",
                "Índice",
                "Prioridade"
            ]

            st.dataframe(
                tabela,
                width="stretch",
                hide_index=True
            )

        # Decisão
        critica = (prio_df["prioridade"] == "Crítica").sum()
        alta = (prio_df["prioridade"] == "Alta").sum()
        media = (prio_df["prioridade"] == "Média").sum()

        secao("Decisão Recomendada")

        if critica > 0:
            mensagem = f"Priorizar imediatamente os **{critica} conteúdos críticos**."
            cor = "red"

        elif alta > 0:
            mensagem = f"Programar a revisão dos **{alta} conteúdos de alta prioridade**."
            cor = "orange"

        elif media > 0:
            mensagem = f"Manter acompanhamento dos **{media} conteúdos de prioridade média**."
            cor = "purple"

        else:
            mensagem = "Não foram identificados conteúdos com prioridade alta ou crítica."
            cor = "green"

        render_html(f"""
            <div class="insight-card insight-{cor}">
                <div class="insight-title">
                    Direcionamento do SAD
                </div>
                <div class="insight-text">
                    {mensagem}
                </div>
            </div>
        """)

render_html("""
    <div class="footer">
        Duolingo Learning Insights — Sistema de Apoio à Decisão<br>
        Ana Leticia · Denise Matos · Lana Liz
    </div>
""")