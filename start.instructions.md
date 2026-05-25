# Instrução para o Agente: Implementação do Dashboard de Fadiga de Campanhas

## Objetivo
Implementar um dashboard web local (Streamlit) para análise de fadiga de campanhas de anúncios (Meta Ads). O sistema deve ser 100% local, lendo arquivos Parquet brutos existentes no computador do usuário, sem necessidade de banco de dados SQL externo ou envio de dados para a nuvem.

## Estrutura de Diretórios
O agente deve implementar a seguinte estrutura de arquivos:

```text
meu-projeto-fadiga/
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
├── app/
│   ├── app.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_engine.py
│   └── models/
│       ├── __init__.py
│       ├── modelo_fadiga.joblib
│       └── predictor.py
└── data/
    └── raw_campaigns/

```

Requisitos Técnicos
A. Segurança e Isolamento

    Zero Internet: A aplicação deve rodar de forma completamente offline. Não deve haver chamadas externas (APIs, telemetria ou bibliotecas que exijam autenticação externa) durante o processamento.

    Proteção de Dados: O arquivo .gitignore deve garantir que as pastas data/ (arquivos .parquet) e os arquivos de modelo (.joblib) sejam ignorados pelo versionamento Git.

B. Engine de Dados e Performance (DuckDB)

    Utilizar duckdb para realizar consultas "in-place" na pasta data/raw_campaigns/*.parquet.

    Implementar em app/utils/data_engine.py:

        get_all_ad_accounts(): SELECT DISTINCT adAccount_id, adAccount_name nos arquivos.

        get_campaigns_by_account(account_id): SELECT DISTINCT campaign_id, campaign_name filtrando pelo adAccount_id.

        load_raw_campaign_history(campaign_id): Filtrar e ordenar o histórico completo da campanha pelo campo context_timestamp.

C. Interface (Streamlit)

    Implementar app.py com:

        Sidebar: selectbox para ad_account e campaign.

        Cache Inteligente: @st.cache_data para evitar varredura desnecessária nos arquivos Parquet.

        Visualização: Gráficos interativos com plotly.express (line chart) para a evolução da fadiga temporal.

        Transparência: st.expander para exibir os dados brutos de forma tabulada (st.dataframe).

D. Integração de Machine Learning

    Em app/models/predictor.py: carregar o modelo_fadiga.joblib via joblib.

    Função de predição: deve receber o dataframe, aplicar modelo.predict() e retornar o conjunto de dados enriquecido com a coluna de fadiga.

E. Containerização (Docker)

    Dockerfile baseado em python:3.11-slim.

    docker-compose.yml configurado para mapear o volume local ./data:/app/data, permitindo que o container acesse os arquivos Parquet da máquina host sem duplicação.

Instruções Adicionais para o Agente

    Priorizar a performance de leitura colunar do DuckDB.

    Garantir que as colunas (adAccount_id, adAccount_name, campaign_id, context_timestamp) sejam processadas corretamente.

    O agente deve solicitar a confirmação do usuário antes de criar os arquivos de configuração (Docker/Git/Requirements).