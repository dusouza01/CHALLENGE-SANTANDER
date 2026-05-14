# Santander Challenge - Análise Inteligente de Empresas PJ

Este projeto é uma solução desenvolvida para o Santander Challenge, focada em analisar e classificar empresas (PJ) com base em seu momento de vida e em suas relações financeiras dentro da rede de transações.

A solução combina análise de dados, machine learning, análise de redes financeiras e visualização interativa através de um dashboard em Streamlit.

---

# 🚀 Tecnologias Utilizadas

- **Linguagem:** Python 3
- **Análise de Dados:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Análise de Rede:** NetworkX
- **Dashboard:** Streamlit
- **Visualização de Dados:** Plotly Express
- **Utilitários:** Tqdm

---

# ✨ Funcionalidades

## 📊 Classificação de Momento de Vida
Utiliza um modelo de clustering (**K-Means**) para categorizar empresas em perfis como:

- Pequeno Porte
- Em Crescimento
- Consolidada
- Grande Porte

---

## 🔗 Análise de Rede Financeira
Mapeia as transações financeiras para:

- Identificar empresas centrais (**hubs**)
- Medir dependência entre parceiros comerciais
- Entender conexões estratégicas dentro da rede

---

## 📈 Projeções Simplificadas
Gera projeções de recebimentos para o próximo mês com base no histórico transacional da empresa.

---

## 🖥️ Dashboard Interativo

O dashboard possui duas visões principais:

### 📌 Visão Geral do Portfólio
- KPIs agregados
- Gráficos interativos
- Filtros dinâmicos
- Distribuição de empresas por perfil

### 🔍 Análise Individual de Empresa
Permite buscar empresas por ID e visualizar:

- Perfil completo
- Métricas financeiras
- Métricas de rede
- Projeções futuras

---

# 📁 Estrutura do Projeto

```bash
challenge_santander/
│
├── analise_completa.py
├── dashboard.py
├── Base 1 - ID.csv
├── Base 2 - Transações.csv
│
├── empresas_analisadas.csv
├── transacoes_com_data.csv
├── analise.log
│
└── README.md
```

---

# ⚙️ Pré-requisitos

Antes de começar, certifique-se de ter instalado:

- Python 3.8 ou superior

---

# 📦 Instalação das Dependências

Abra o terminal na pasta do projeto e execute o comando abaixo para instalar todas as bibliotecas necessárias:

```bash
pip install pandas scikit-learn networkx streamlit plotly tqdm
```

---

# 🚀 Como Executar o Projeto

A aplicação funciona em duas etapas simples.

---

## 🔹 Etapa 1 — Processar os Dados

Execute o script principal de análise:

```bash
python analise_completa.py
```

Este script irá:

- Ler os arquivos:
  - `Base 1 - ID.csv`
  - `Base 2 - Transações.csv`

- Realizar:
  - Tratamento dos dados
  - Feature engineering
  - Modelagem
  - Análise de rede financeira

- Gerar os arquivos:
  - `empresas_analisadas.csv`
  - `transacoes_com_data.csv`

- Salvar um log detalhado da execução:
  - `analise.log`

---

## 🔹 Etapa 2 — Iniciar o Dashboard

Após processar os dados, execute:

```bash
streamlit run dashboard.py
```

O Streamlit abrirá automaticamente uma nova aba no navegador com o dashboard interativo.

---

# 📊 Resultado Esperado

Após a execução completa, o projeto permitirá:

- Visualizar métricas do portfólio
- Explorar empresas individualmente
- Entender conexões financeiras
- Identificar empresas estratégicas
- Obter insights de crescimento e dependência financeira

---

# 📌 Objetivo do Projeto

O principal objetivo desta solução é auxiliar na tomada de decisão estratégica através da análise inteligente de empresas PJ, utilizando ciência de dados e análise de redes financeiras.

---

# 👨‍💻 Autor

Projeto desenvolvido para o Santander Challenge.
