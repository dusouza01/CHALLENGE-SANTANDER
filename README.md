# Santander Challenge — Análise Inteligente de Empresas PJ

Solução desenvolvida para o **Santander Challenge**, focada em analisar e classificar empresas (PJ) com base em seu momento de vida e em suas relações financeiras dentro da rede de transações.

A solução combina análise de dados, machine learning, análise de redes financeiras e visualização interativa por meio de um dashboard em Streamlit.

---

## 🚀 Tecnologias Utilizadas

| Categoria | Tecnologia |
|---|---|
| Linguagem | Python 3 |
| Análise de Dados | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Análise de Rede | NetworkX |
| Dashboard | Streamlit |
| Visualização | Plotly Express |
| Utilitários | Tqdm |

---

## ✨ Funcionalidades

### 📊 Classificação de Momento de Vida

Utiliza um modelo de clustering (**K-Means**) para categorizar empresas em perfis:

- Pequeno Porte
- Em Crescimento
- Consolidada
- Grande Porte

### 🔗 Análise de Rede Financeira

Mapeia as transações financeiras para:

- Identificar empresas centrais (**hubs**)
- Medir dependência entre parceiros comerciais
- Entender conexões estratégicas dentro da rede

### 📈 Projeções Simplificadas

Gera projeções de recebimentos para o próximo mês com base no histórico transacional da empresa.

### 🖥️ Dashboard Interativo

O dashboard possui duas visões principais:

**Visão Geral do Portfólio**
- KPIs agregados
- Gráficos interativos com filtros dinâmicos
- Distribuição de empresas por perfil

**Análise Individual de Empresa**
- Busca por ID
- Perfil completo com métricas financeiras e de rede
- Projeções futuras

---

## 📁 Estrutura do Projeto

```
challenge_santander/
│
├── analise_completa.py       # Script principal de análise
├── dashboard.py              # Dashboard interativo
├── Base 1 - ID.csv           # Base de dados de identificação
├── Base 2 - Transações.csv   # Base de dados de transações
│
├── empresas_analisadas.csv   # Saída: empresas processadas
├── transacoes_com_data.csv   # Saída: transações tratadas
├── analise.log               # Log detalhado da execução
│
└── README.md
```

---

## ⚙️ Pré-requisitos

- Python 3.8 ou superior

---

## 📦 Instalação

Na pasta do projeto, execute:

```bash
pip install pandas scikit-learn networkx streamlit plotly tqdm
```

---

## ▶️ Como Executar

### Etapa 1 — Processar os Dados

```bash
python analise_completa.py
```

O script irá:
1. Ler `Base 1 - ID.csv` e `Base 2 - Transações.csv`
2. Realizar tratamento de dados, feature engineering, modelagem e análise de rede
3. Gerar `empresas_analisadas.csv` e `transacoes_com_data.csv`
4. Salvar o log em `analise.log`

### Etapa 2 — Iniciar o Dashboard

```bash
streamlit run dashboard.py
```

O Streamlit abrirá automaticamente uma aba no navegador com o dashboard interativo.

---

## 📊 Resultados

Após a execução completa, o projeto permitirá:

- Visualizar métricas agregadas do portfólio
- Explorar empresas individualmente por ID
- Entender conexões e dependências financeiras
- Identificar empresas estratégicas na rede
- Obter insights de crescimento e risco financeiro

---

## 🎯 Objetivo

Auxiliar na **tomada de decisão estratégica** por meio da análise inteligente de empresas PJ, aplicando ciência de dados e análise de redes financeiras.

---

## 👨‍💻 Autor

Projeto desenvolvido para o **Santander Challenge**.
