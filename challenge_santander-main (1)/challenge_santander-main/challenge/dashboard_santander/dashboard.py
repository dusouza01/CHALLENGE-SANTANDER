import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
from pyvis.network import Network # Importação para o gráfico de rede
import streamlit.components.v1 as components

# --- Configuração da Página ---
st.set_page_config(page_title="Sistema de Análise PJ", layout="wide", initial_sidebar_state="expanded")

# --- LÓGICA DE AUTENTICAÇÃO ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if st.session_state["authenticated"]:
        return True
    st.title("Sistema de Análise e Prospecção PJ")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.header("Por favor, faça o login para continuar")
            username = st.text_input("Usuário")
            password = st.text_input("Senha", type="password")
            submitted = st.form_submit_button("Entrar")
            if submitted:
                if username == "admin" and password == "admin":
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Usuário ou senha inválidos")
    return False

if not check_password():
    st.stop()
# --- FIM DA LÓGICA DE AUTENTICAÇÃO ---

# --- Carregamento e Preparação dos Dados e Modelos ---
@st.cache_data
def carregar_dados_e_modelos():
    try:
        df = pd.read_csv("empresas_analisadas.csv")
        df['DT_REFE'] = pd.to_datetime(df['DT_REFE'])
        df_transacoes = pd.read_csv("transacoes_com_data.csv")
        kmeans_model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        model_features = joblib.load('model_features.pkl')
        cluster_mapping = joblib.load('cluster_mapping.pkl')
    except FileNotFoundError as e:
        st.error(f"Erro ao carregar arquivos necessários. Verifique se todos os arquivos (.csv, .pkl) estão na pasta. Detalhe: {e}")
        return None, None, None, None, None, None
    return df, df_transacoes, kmeans_model, scaler, model_features, cluster_mapping

df, df_transacoes, kmeans_model, scaler, model_features, cluster_mapping = carregar_dados_e_modelos()

if df is None:
    st.stop()

# --- Barra Lateral ---
st.sidebar.title("Sistema de Análise PJ")
pagina_selecionada = st.sidebar.radio(
    "Navegar", 
    ["Visão Geral e Prospecção", "Análise de Risco", "Análise Individual", "Análise de Rede", "Insights Comerciais", "Simulador de Cenários"]
)
st.sidebar.markdown("---")
if st.sidebar.button("Sair"):
    st.session_state["authenticated"] = False
    st.rerun()

# ==============================================================================
# --- PÁGINA 1: VISÃO GERAL E PROSPECÇÃO ---
# ==============================================================================
if pagina_selecionada == "Visão Geral e Prospecção":
    st.title("Dashboard de Prospecção e Análise de Empresas")
    
    st.sidebar.header("Filtros de Segmentação")
    momentos = df['MOMENTO_VIDA'].unique()
    momento_selecionado = st.sidebar.multiselect("Momento de Vida:", options=momentos, default=momentos)
    faturamento_selecionado = st.sidebar.slider("Faixa de Faturamento Anual (R$):", int(df['VL_FATU'].min()), int(df['VL_FATU'].max()), (int(df['VL_FATU'].min()), int(df['VL_FATU'].max())))
    
    df_filtrado = df[
        (df['MOMENTO_VIDA'].isin(momento_selecionado)) & 
        (df['VL_FATU'].between(*faturamento_selecionado))
    ].copy()
    
    total_empresas = df_filtrado['ID'].nunique()
    faturamento_medio = df_filtrado['VL_FATU'].mean()
    idade_media = df_filtrado['IDADE_EMPRESA'].mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Empresas Únicas", f"{total_empresas:,}".replace(",", "."))
    col2.metric("Faturamento Médio (por registro)", f"R$ {faturamento_medio:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col3.metric("Idade Média (por registro)", f"{idade_media:.1f} anos")
    st.markdown("---")
    
    col_graf1, col_graf2 = st.columns(2)
    with col_graf1:
        st.subheader("Distribuição por Momento de Vida")
        fig_bar = px.bar(df_filtrado['MOMENTO_VIDA'].value_counts().reset_index(), x='MOMENTO_VIDA', y='count', color='MOMENTO_VIDA', text_auto=True, labels={'count': 'Registros', 'MOMENTO_VIDA': 'Momento de Vida'})
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_graf2:
        st.subheader("Faturamento vs. Idade")
        if not df_filtrado.empty:
            amostra_grafico = df_filtrado.sample(min(1000, len(df_filtrado))).copy()
            amostra_grafico['VL_SLDO_TAMANHO'] = amostra_grafico['VL_SLDO'] - amostra_grafico['VL_SLDO'].min()
            fig_scatter = px.scatter(amostra_grafico, x='IDADE_EMPRESA', y='VL_FATU', color='MOMENTO_VIDA', hover_name='ID', size='VL_SLDO_TAMANHO', hover_data={'VL_SLDO': ':,.2f'}, labels={'IDADE_EMPRESA': 'Idade (Anos)', 'VL_FATU': 'Faturamento (R$)'})
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Não há dados para exibir nos gráficos.")
            
    with st.expander("Ver Validação Visual dos Grupos de 'Momento de Vida'"):
        st.subheader("Distribuição de Métricas por Momento de Vida")
        if not df_filtrado.empty:
            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                fig_fatu = px.box(df_filtrado, x='MOMENTO_VIDA', y='VL_FATU', color='MOMENTO_VIDA', title="Faturamento Anual")
                st.plotly_chart(fig_fatu, use_container_width=True)
            with col_v2:
                fig_saldo = px.box(df_filtrado, x='MOMENTO_VIDA', y='VL_SLDO', color='MOMENTO_VIDA', title="Saldo em Conta")
                st.plotly_chart(fig_saldo, use_container_width=True)
            with col_v3:
                fig_idade = px.box(df_filtrado, x='MOMENTO_VIDA', y='IDADE_EMPRESA', color='MOMENTO_VIDA', title="Idade da Empresa")
                st.plotly_chart(fig_idade, use_container_width=True)
        else:
            st.warning("Não há dados filtrados para exibir a validação.")
            
    with st.expander("Ver Amostra dos Dados Filtrados"):
        st.dataframe(df_filtrado[['ID', 'DT_REFE', 'MOMENTO_VIDA', 'VL_FATU', 'IDADE_EMPRESA', 'DS_CNAE']].head(1000).style.format({'VL_FATU': 'R$ {:,.2f}'}))
        st.caption(f"Exibindo as primeiras 1.000 de {len(df_filtrado)} registros encontrados.")


# ==============================================================================
# --- PÁGINA 2: ANÁLISE DE RISCO ---
# ==============================================================================
elif pagina_selecionada == "Análise de Risco":
    st.title("Painel de Gestão de Risco da Carteira")
    
    st.sidebar.header("Filtros de Risco")
    risco_selecionado = st.sidebar.slider(
        "Filtrar por Score de Risco:",
        min_value=0, max_value=100, value=(0, 100)
    )
    
    df_filtrado_risco = df[df['SCORE_RISCO'].between(*risco_selecionado)]
    
    st.subheader("Alertas e Saúde Geral da Carteira")
    df_recente = df_filtrado_risco.sort_values('DT_REFE').drop_duplicates('ID', keep='last')
    total_empresas_unicas = df_recente['ID'].nunique()
    risco_alto = df_recente[df_recente['SCORE_RISCO'] > 75]['ID'].nunique()
    risco_medio = df_recente[(df_recente['SCORE_RISCO'] > 40) & (df_recente['SCORE_RISCO'] <= 75)]['ID'].nunique()
    risco_baixo = df_recente[df_recente['SCORE_RISCO'] <= 40]['ID'].nunique()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Empresas na Seleção", f"{total_empresas_unicas}")
    col2.metric("Risco Alto (> 75)", f"{risco_alto}", delta=f"{((risco_alto/total_empresas_unicas)*100):.1f}%" if total_empresas_unicas > 0 else "0.0%", delta_color="inverse")
    col3.metric("Risco Médio (41-75)", f"{risco_medio}", delta=f"{((risco_medio/total_empresas_unicas)*100):.1f}%" if total_empresas_unicas > 0 else "0.0%", delta_color="off")
    col4.metric("Risco Baixo (0-40)", f"{risco_baixo}", delta=f"{((risco_baixo/total_empresas_unicas)*100):.1f}%" if total_empresas_unicas > 0 else "0.0%", delta_color="normal")
    st.markdown("---")
    
    st.subheader("Diagnóstico de Risco Individual")
    id_unicos_risco = sorted(df_recente['ID'].unique())
    id_pesquisado_risco = st.selectbox("Selecione uma Empresa para análise de risco:", options=id_unicos_risco)
    
    if id_pesquisado_risco:
        ultimo_registro_risco = df_recente[df_recente['ID'] == id_pesquisado_risco].iloc[0]
        score_risco_atual = ultimo_registro_risco['SCORE_RISCO']
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score_risco_atual,
            title = {'text': f"Score de Risco para {id_pesquisado_risco}"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "black"},
                'steps' : [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "red"}],
            }))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)


# ==============================================================================
# --- PÁGINA 3: ANÁLISE INDIVIDUAL ---
# ==============================================================================
elif pagina_selecionada == "Análise Individual":
    st.title("Análise Individual da Empresa")
    id_unicos = sorted(df['ID'].unique())
    id_pesquisado = st.selectbox("Selecione o ID da Empresa:", options=id_unicos)

    if id_pesquisado:
        dados_empresa = df[df['ID'] == id_pesquisado].sort_values('DT_REFE')
        ultimo_registro = dados_empresa.iloc[-1]
        
        st.header(f"Resultados para: {ultimo_registro['ID']}")
        st.markdown(f"**CNAE:** {ultimo_registro['DS_CNAE']}")
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["Visão Geral e Histórico", "Comparação com Pares"])
        
        with tab1:
            st.subheader("Métricas Chave do Período")
            col1, col2, col3 = st.columns(3)
            col1.metric("Faturamento Anual (constante)", f"R$ {ultimo_registro['VL_FATU']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            col2.metric("Saldo Médio", f"R$ {dados_empresa['VL_SLDO'].mean():,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            col3.metric("Projeção Próximo Mês", f"R$ {ultimo_registro['PROJECAO_RECEBIMENTO']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            st.subheader("Evolução Mensal do Saldo em Conta")
            fig_saldo = px.line(dados_empresa, x='DT_REFE', y='VL_SLDO', markers=True, labels={'VL_SLDO': 'Saldo (R$)', 'DT_REFE': 'Data'}, title="Variação do Saldo em Conta")
            st.plotly_chart(fig_saldo, use_container_width=True)
            with st.expander("Ver Registros Históricos Detalhados"):
                st.dataframe(dados_empresa[['DT_REFE', 'MOMENTO_VIDA', 'VL_FATU', 'VL_SLDO', 'IDADE_EMPRESA', 'SCORE_RISCO']].style.format({'VL_FATU': 'R$ {:,.2f}', 'VL_SLDO': 'R$ {:,.2f}', 'IDADE_EMPRESA': '{:.1f} anos', 'SCORE_RISCO': '{:d}'}))

        with tab2:
            st.subheader("Benchmarking de Performance vs. Pares do Setor")
            momento_atual = ultimo_registro['MOMENTO_VIDA']
            cnae_atual = ultimo_registro['DS_CNAE']
            df_peers = df[(df['DS_CNAE'] == cnae_atual) & (df['MOMENTO_VIDA'] == momento_atual) & (df['ID'] != id_pesquisado)]
            if df_peers.empty:
                st.warning("Não foram encontrados pares com o mesmo CNAE e Momento de Vida para comparação.")
            else:
                peer_avg_fatu = df_peers['VL_FATU'].mean()
                peer_avg_saldo = df_peers['VL_SLDO'].mean()
                delta_fatu_str, delta_saldo_str = "N/A", "N/A"
                if peer_avg_fatu > 0:
                    delta_fatu_val = ((ultimo_registro['VL_FATU'] - peer_avg_fatu) / peer_avg_fatu) * 100
                    delta_fatu_str = f"{delta_fatu_val:.1f}%"
                if peer_avg_saldo != 0:
                    delta_saldo_val = ((dados_empresa['VL_SLDO'].mean() - peer_avg_saldo) / abs(peer_avg_saldo)) * 100
                    delta_saldo_str = f"{delta_saldo_val:.1f}%"
                col1, col2 = st.columns(2)
                col1.metric(f"Faturamento Anual ({momento_atual})", f"R$ {ultimo_registro['VL_FATU']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), delta=delta_fatu_str)
                col2.metric("Saldo Médio (no período)", f"R$ {dados_empresa['VL_SLDO'].mean():,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), delta=delta_saldo_str)
                st.caption(f"A comparação é feita com {len(df_peers['ID'].unique())} empresas do setor '{cnae_atual}' classificadas como '{momento_atual}'.")

# ==============================================================================
# --- PÁGINA 4: ANÁLISE DE REDE ---
# ==============================================================================
elif pagina_selecionada == "Análise de Rede":
    st.title("Mapa Interativo da Rede de Relacionamentos")
    st.markdown("Selecione uma empresa para visualizar seus principais parceiros comerciais (clientes e fornecedores) com base no volume total de transações.")

    id_unicos_rede = sorted(df['ID'].unique())
    id_pesquisado_rede = st.selectbox("Selecione a Empresa para visualizar a rede:", options=id_unicos_rede)
    
    top_n = st.slider("Número de principais parceiros a exibir:", min_value=1, max_value=10, value=5)

    if id_pesquisado_rede:
        pagamentos_recebidos = df_transacoes[df_transacoes['ID_RCBE'] == id_pesquisado_rede]
        pagamentos_feitos = df_transacoes[df_transacoes['ID_PGTO'] == id_pesquisado_rede]
        top_clientes = pagamentos_recebidos.groupby('ID_PGTO')['VL'].sum().nlargest(top_n)
        top_fornecedores = pagamentos_feitos.groupby('ID_RCBE')['VL'].sum().nlargest(top_n)
        
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True, directed=True)
        net.add_node(id_pesquisado_rede, label=id_pesquisado_rede, color='#D32F2F', size=30, title=f"Empresa Analisada\nID: {id_pesquisado_rede}")
        
        for cliente, valor in top_clientes.items():
            net.add_node(cliente, label=cliente, color='#4CAF50', size=15, title=f"Cliente\nID: {cliente}\nVolume: R$ {valor:,.2f}")
            net.add_edge(cliente, id_pesquisado_rede, value=valor, title=f"R$ {valor:,.2f}")
            
        for fornecedor, valor in top_fornecedores.items():
            net.add_node(fornecedor, label=fornecedor, color='#FFC107', size=15, title=f"Fornecedor\nID: {fornecedor}\nVolume: R$ {valor:,.2f}")
            net.add_edge(id_pesquisado_rede, fornecedor, value=valor, title=f"R$ {valor:,.2f}")

        try:
            net.save_graph('network.html')
            HtmlFile = open('network.html', 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
            components.html(source_code, height=800)
        except Exception as e:
            st.error(f"Ocorreu um erro ao gerar o gráfico de rede: {e}")

# ==============================================================================
# --- PÁGINA 5: INSIGHTS COMERCIAIS ---
# ==============================================================================
elif pagina_selecionada == "Insights Comerciais":
    st.title("Insights para Abordagem Comercial")
    id_unicos = sorted(df['ID'].unique())
    id_pesquisado = st.selectbox("Selecione o ID da Empresa para gerar insights:", options=id_unicos)

    if id_pesquisado:
        dados_empresa = df[df['ID'] == id_pesquisado].sort_values('DT_REFE')
        ultimo_registro = dados_empresa.iloc[-1]
        st.header(f"Sugestões de Abordagem para: {ultimo_registro['ID']}")
        st.markdown(f"**CNAE:** {ultimo_registro['DS_CNAE']} | **Momento de Vida Atual:** {ultimo_registro['MOMENTO_VIDA']}")
        def gerar_insights(empresa, historico):
            insights = []
            saldo_medio = historico['VL_SLDO'].mean()
            if saldo_medio < 0:
                insights.append(f"**Ponto de Atenção: Fluxo de Caixa Negativo**\n\n- O saldo médio da empresa nos últimos meses foi de **R$ {saldo_medio:,.2f}**. Isso indica uma forte necessidade de capital de giro.\n\n- **Produto Sugerido:** Oferta proativa de **Capital de Giro** com taxas competitivas.")
            elif saldo_medio < (empresa['VL_FATU'] / 12 * 0.5):
                insights.append(f"**Oportunidade: Otimização de Caixa**\n\n- A empresa opera com um saldo médio de **R$ {saldo_medio:,.2f}**, que é relativamente baixo para seu faturamento.\n\n- **Produto Sugerido:** Apresentar soluções de **Gestão de Caixa** e **Investimentos de Curto Prazo**.")
            if empresa['NIVEL_DEPENDENCIA'] > 0.4:
                insights.append(f"**Risco de Concentração**\n\n- **{empresa['NIVEL_DEPENDENCIA']:.1%}** das transações da empresa estão concentradas em um único parceiro comercial.\n\n- **Argumento de Venda:** Posicionar o banco como um parceiro estratégico para mitigar riscos, oferecendo **Seguro de Crédito**.")
            if empresa['CENTRALIDADE'] > df['CENTRALIDADE'].quantile(0.85):
                insights.append(f"**Oportunidade Estratégica: Empresa-Hub**\n\n- Esta empresa é um **hub** em sua rede, com alta conectividade.\n\n- **Produto Sugerido:** Oferta de **Plataforma de Pagamentos Automatizados**.")
            if empresa['MOMENTO_VIDA'] == 'Em Crescimento':
                insights.append(f"**Apoio ao Crescimento**\n\n- A empresa está classificada como 'Em Crescimento', indicando uma fase de expansão.\n\n- **Produto Sugerido:** Linhas de crédito para **investimento em ativos (FINAME, BNDES)**.")
            if not insights:
                insights.append("**Perfil Estável**\n\n- A empresa apresenta um perfil financeiro estável. O foco deve ser no **relacionamento e na oferta de produtos que superem a concorrência**.")
            return insights
        st.markdown("---")
        lista_insights = gerar_insights(ultimo_registro, dados_empresa)
        for insight in lista_insights:
            st.info(insight)

# ==============================================================================
# --- PÁGINA 6: SIMULADOR DE CENÁRIOS ---
# ==============================================================================
elif pagina_selecionada == "Simulador de Cenários":
    st.title("Simulador de Cenários Futuros")
    st.markdown("Esta ferramenta permite simular a evolução de uma empresa com base em premissas de crescimento, projetando seu Faturamento, Saldo e seu **Momento de Vida**.")
    
    id_unicos = sorted(df['ID'].unique())
    id_pesquisado = st.selectbox("Selecione a Empresa para simular:", options=id_unicos)
    
    if id_pesquisado:
        st.markdown("---")
        st.subheader("Parâmetros da Simulação")
        
        col1, col2 = st.columns(2)
        with col1:
            taxa_crescimento_anual = st.slider("Taxa de Crescimento Anual do Faturamento (%):", min_value=-10, max_value=50, value=10, step=1)
        with col2:
            horizonte_anos = st.slider("Horizonte da Simulação (Anos):", min_value=1, max_value=5, value=5, step=1)
            
        if st.button("Executar Simulação"):
            dados_atuais = df[df['ID'] == id_pesquisado].sort_values('DT_REFE').iloc[-1]
            resultados_simulacao = []
            
            for ano in range(1, horizonte_anos + 1):
                dados_simulados = dados_atuais.copy()
                dados_simulados['VL_FATU'] = dados_atuais['VL_FATU'] * ((1 + taxa_crescimento_anual / 100) ** ano)
                if dados_atuais['VL_FATU'] > 0:
                    ratio = dados_simulados['VL_FATU'] / dados_atuais['VL_FATU']
                    dados_simulados['VL_SLDO'] = dados_atuais['VL_SLDO'] * ratio
                    dados_simulados['VL_PAGAMENTOS'] = dados_atuais['VL_PAGAMENTOS'] * ratio
                    dados_simulados['VL_RECEBIMENTOS'] = dados_atuais['VL_RECEBIMENTOS'] * ratio
                dados_simulados['IDADE_EMPRESA'] = dados_atuais['IDADE_EMPRESA'] + ano
                dados_simulados['Ano'] = f"Ano {ano}"
                resultados_simulacao.append(dados_simulados)

            df_simulado = pd.DataFrame(resultados_simulacao)
            
            X_simulado = df_simulado[model_features]
            X_simulado_scaled = scaler.transform(X_simulado)
            cluster_predito = kmeans_model.predict(X_simulado_scaled)
            df_simulado['Momento de Vida Projetado'] = cluster_predito
            df_simulado['Momento de Vida Projetado'] = df_simulado['Momento de Vida Projetado'].map(cluster_mapping)
            
            st.markdown("---")
            st.subheader("Resultados da Simulação")

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_fatu = px.line(df_simulado, x='Ano', y='VL_FATU', markers=True, title="Projeção do Faturamento Anual")
                fig_fatu.update_layout(yaxis_title="Faturamento (R$)")
                st.plotly_chart(fig_fatu, use_container_width=True)
            with col_g2:
                fig_saldo = px.line(df_simulado, x='Ano', y='VL_SLDO', markers=True, title="Projeção do Saldo em Conta")
                fig_saldo.update_layout(yaxis_title="Saldo (R$)")
                st.plotly_chart(fig_saldo, use_container_width=True)

            st.subheader("Evolução Projetada")
            st.dataframe(df_simulado[['Ano', 'VL_FATU', 'VL_SLDO', 'IDADE_EMPRESA', 'Momento de Vida Projetado']].style.format({
                'VL_FATU': 'R$ {:,.2f}',
                'VL_SLDO': 'R$ {:,.2f}',
                'IDADE_EMPRESA': '{:.1f} anos'
            }))