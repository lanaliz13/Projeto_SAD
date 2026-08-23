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


def explicacao_grafico(titulo, texto):
    render_html(f"""
        <div class="insight-card insight-purple" style="margin-bottom: 1.2rem;">
            <div class="insight-title">💡 Como interpretar este gráfico ({html.escape(titulo)}):</div>
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
            "Simulador de Prioridade",
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
        "A retenção cai à medida que os dias passam sem revisão. O sistema identifica automaticamente as faixas onde a retenção cai abaixo do limite crítico para que a intervenção seja imediata."
    )

    c_esq1, c_esq2 = st.columns(2)
    with c_esq1:
        idioma = st.selectbox("Idioma para análise", ["Todos"] + idiomas, key="esq_lang")
    with c_esq2:
        dias_corte = st.slider("Dias sem prática para alerta crítico", 7, 60, 14, key="esq_dias")

    df = filtrar_idioma(curve, idioma) if idioma != "Todos" else curve
    lag_col = obter_coluna(df, ["lag_bin", "lag_days", "avg_lag_days"])
    recall_col = obter_recall_col(df)

    if not df.empty and lag_col and recall_col:
        res = df.groupby(lag_col, as_index=False, observed=False)[recall_col].mean()

        res["status_risco"] = res[recall_col].apply(
            lambda r: "Crítico" if r < limite_critico else ("Atenção" if r < meta_recall else "Seguro")
        )

        qtd_criticos = len(res[res["status_risco"] == "Crítico"])
        qtd_atencao = len(res[res["status_risco"] == "Atenção"])
        retencao_minima = res[recall_col].min()

        m1, m2, m3 = st.columns(3)
        with m1:
            caixa_metrica("Faixas Críticas", str(qtd_criticos), "Abaixo do limite crítico")
        with m2:
            caixa_metrica("Faixas em Atenção", str(qtd_atencao), "Abaixo da meta global")
        with m3:
            caixa_metrica("Pior Retenção Observada", percentual(retencao_minima), "Menor taxa de acerto")

        fig = px.bar(
            res, x=lag_col, y=recall_col, text_auto='.1%',
            color="status_risco",
            color_discrete_map={"Crítico": "#ff4d67", "Atenção": "#f59e0b", "Seguro": "#34d399"},
            title=f"Taxa de Retenção por Intervalo de Estudo — {idioma}",
            template="plotly_dark"
        )
        fig.add_hline(y=meta_recall, line_dash="dash", line_color="#34d399", annotation_text="Meta (85%)")
        fig.add_hline(y=limite_critico, line_dash="dot", line_color="#fb7185", annotation_text="Limite Crítico")
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis_tickformat='.0%', yaxis_title="Taxa de Acerto", xaxis_title="Intervalo Sem Prática"
        )
        st.plotly_chart(fig, width="stretch")

        if qtd_criticos > 0:
            render_html(f"""
                <div class="insight-card insight-red">
                    <div class="insight-title">⚠️ Diagnóstico Crítico de Esquecimento</div>
                    <div class="insight-text">
                        Foram identificadas <b>{qtd_criticos} faixas temporais na zona crítica de esquecimento</b> (retenção menor que {limite_critico*100:.0f}%). 
                        Recomenda-se acionar revisões reforçadas para alunos com mais de {dias_corte} dias sem prática.
                    </div>
                </div>
            """)
        else:
            render_html(f"""
                <div class="insight-card insight-green">
                    <div class="insight-title">✅ Estabilidade de Retenção</div>
                    <div class="insight-text">
                        Nenhum intervalo sem prática atingiu o nível crítico para este filtro. A retenção média continua dentro dos limites toleráveis.
                    </div>
                </div>
            """)
    else:
        st.info("Registros insuficientes para a Análise de Esquecimento neste filtro.")

# --- REVISÕES ---
elif pagina == "Revisões":
    cabecalho("Análise Avançada de Revisões", "Comportamento da retenção sob diferentes frequências de repetição.")

    explicacao_grafico(
        "Curva de Aprendizado por Prática",
        "A linha mostra que a curva de retenção sobe à medida que o histórico de treinos aumenta. Isso comprova a eficácia da repetição espaçada no aprendizado."
    )

    idioma = st.selectbox("Idioma", ["Todos"] + idiomas)
    df = filtrar_idioma(curve, idioma) if idioma != "Todos" else curve

    exp_col = obter_coluna(df, ["practice_bin", "avg_prior_exposures", "prior_exposures"])
    recall_col = obter_recall_col(df)

    if not df.empty and exp_col and recall_col:
        res = df.groupby(exp_col, as_index=False, observed=False)[recall_col].mean()
        fig = px.line(
            res, x=exp_col, y=recall_col, markers=True,
            title=f"Evolução da Retenção por Histórico de Treino — {idioma}",
            color_discrete_sequence=['#8b5cf6']
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=10))
        fig.update_layout(
            title=f"Evolução da Retenção por Histórico de Treino — {idioma}",
            xaxis_title="Faixa de Prática / Exposições", yaxis_title="Recall Médio",
            yaxis_tickformat='.0%', template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Registros insuficientes para a Análise de Revisões neste filtro.")

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

# --- SIMULADOR DE PRIORIDADE ---
elif pagina == "Simulador de Prioridade":
    cabecalho("Simulador de Prioridade de Revisão", "Calcule em tempo real o risco e o índice de prioridade de revisão.")

    col1, col2 = st.columns(2)
    with col1:
        idioma = st.selectbox("Idioma", ["Todos"] + idiomas, key="sim_lang")
        dias_sem_pratica = st.number_input("Dias sem prática", 0, 365, 30)
    with col2:
        exposicoes = st.number_input("Exposições anteriores", 0, 100, 1)
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
                'bar': {'color': "#ff4d67" if indice > 80 else ("#f59e0b" if indice > 60 else "#8b5cf6")},
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

        rec = obter_recomendacao(prio)
        cor_card = "red" if prio == "Crítica" else ("orange" if prio == "Alta" else ("purple" if prio == "Média" else "green"))
        render_html(f"""
            <div class="insight-card insight-{cor_card}" style="margin-top: 1rem;">
                <div class="insight-title">Direcionamento do Simulador — Prioridade {prio}</div>
                <div class="insight-text">{rec}</div>
            </div>
        """)

# --- DECISÕES ---
elif pagina == "Decisões":
    cabecalho(
        "Painel Executivo de Decisão",
        "Visão estratégica e priorização acionável do vocabulário para intervenção pedagógica."
    )

    c_param1, c_param2, c_param3 = st.columns(3)
    with c_param1:
        tempo_padrao = st.slider("Tempo de referência sem prática (dias)", 1, 60, 14)
    with c_param2:
        exposicoes_padrao = st.slider("Exposições de referência", 0, 20, 5)
    with c_param3:
        idioma = st.selectbox("Idioma para priorização", ["Todos"] + idiomas)

    df_w = filtrar_idioma(words, idioma) if idioma != "Todos" else words
    prio_df = calcular_prioridades_conteudos(df_w, tempo_padrao, exposicoes_padrao)

    if not prio_df.empty:
        critica = len(prio_df[prio_df["prioridade"] == "Crítica"])
        alta = len(prio_df[prio_df["prioridade"] == "Alta"])
        media = len(prio_df[prio_df["prioridade"] == "Média"])
        baixa = len(prio_df[prio_df["prioridade"] == "Baixa"])

        c1, c2, c3, c4 = st.columns(4)
        with c1: caixa_metrica("Prioridade Crítica", str(critica), "Revisão imediata necessária")
        with c2: caixa_metrica("Prioridade Alta", str(alta), "Atenção nas próximas sessões")
        with c3: caixa_metrica("Prioridade Média", str(media), "Acompanhamento de rotina")
        with c4: caixa_metrica("Prioridade Baixa", str(baixa), "Retenção sob controle")

        secao("Visão Geral de Distribuição da Base")

        col_graf1, col_graf2 = st.columns([1, 1])

        with col_graf1:
            counts = prio_df["prioridade"].value_counts().reset_index()
            counts.columns = ["Prioridade", "Quantidade"]

            fig_donut = px.pie(
                counts, names="Prioridade", values="Quantidade",
                title=f"Proporção por Categoria de Risco — {idioma}",
                hole=0.5, color="Prioridade",
                color_discrete_map={"Baixa": "#34d399", "Média": "#f59e0b", "Alta": "#fb7185", "Crítica": "#ff4d67"},
                template="plotly_dark"
            )
            fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_donut, width="stretch")

        with col_graf2:
            fig_hist = px.histogram(
                prio_df, x="indice_prioridade", nbins=20,
                title="Distribuição do Índice de Prioridade de Revisão (0–100)",
                color_discrete_sequence=['#8b5cf6'], template="plotly_dark"
            )
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Índice de Prioridade", yaxis_title="Quantidade de Palavras"
            )
            st.plotly_chart(fig_hist, width="stretch")

        secao(
            f"Top {quantidade_prioridades} Conteúdos com Maior Prioridade de Revisão",
            "Ranking ordenado pelo Índice de Prioridade calculando combinação de recall, tempo e dificuldade."
        )

        palavra_col = obter_nome_palavra(prio_df)
        if palavra_col:
            ranking = prio_df.sort_values("indice_prioridade", ascending=False).head(quantidade_prioridades).copy()

            colunas_exibir = [palavra_col, "idioma", "classe_gramatical", "recall_utilizado", "dificuldade", "indice_prioridade", "prioridade"]
            colunas_existentes = [c for c in colunas_exibir if c in ranking.columns]

            tabela_exibicao = ranking[colunas_existentes].copy()

            if "recall_utilizado" in tabela_exibicao.columns:
                tabela_exibicao["recall_utilizado"] = (tabela_exibicao["recall_utilizado"] * 100).round(1).astype(str) + "%"
            if "dificuldade" in tabela_exibicao.columns:
                tabela_exibicao["dificuldade"] = (tabela_exibicao["dificuldade"] * 100).round(1).astype(str) + "%"
            if "indice_prioridade" in tabela_exibicao.columns:
                tabela_exibicao["indice_prioridade"] = tabela_exibicao["indice_prioridade"].round(1)

            tabela_exibicao.columns = [
                "Palavra / Termo" if c == palavra_col else
                "Idioma" if c == "idioma" else
                "Classe Gramatical" if c == "classe_gramatical" else
                "Recall Observado" if c == "recall_utilizado" else
                "Dificuldade Est." if c == "dificuldade" else
                "Índice de Prioridade" if c == "indice_prioridade" else
                "Prioridade" if c == "prioridade" else c
                for c in tabela_exibicao.columns
            ]

            st.dataframe(tabela_exibicao, width="stretch", hide_index=True)

        if critica > 0:
            rec_texto = f"Existem **{critica} palavras em estado crítico**. A recomendação é inserir esses termos imediatamente nos próximos blocos de prática."
            tipo_rec = "red"
        elif alta > 0:
            rec_texto = f"Existem **{alta} palavras com prioridade alta**. Recomenda-se programar a revisão para os próximos 3 dias."
            tipo_rec = "orange"
        else:
            rec_texto = "A retenção geral da base está dentro dos níveis aceitáveis. Mantenha o fluxo regular de treino."
            tipo_rec = "green"

        secao("Decisão Recomendada pelo SAD")
        render_html(f"""
            <div class="insight-card insight-{tipo_rec}">
                <div class="insight-title">Direcionamento Estratégico</div>
                <div class="insight-text">{rec_texto}</div>
            </div>
        """)

    else:
        st.warning("Não foi possível calcular a distribuição de prioridades para este filtro.")

render_html("""
    <div class="footer">
        Duolingo Learning Insights — Sistema de Apoio à Decisão<br>
        Ana Leticia · Denise Matos · Lana Liz
    </div>
""")