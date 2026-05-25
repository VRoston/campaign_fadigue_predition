# Dashboard Local de Fadiga de Campanhas

Aplicacao local em Streamlit para analisar fadiga de campanhas (Meta Ads) lendo arquivos Parquet em `data/raw_campaigns`.

## Requisitos

- Python 3.11+
- Arquivos Parquet flat em `data/raw_campaigns/`

Colunas esperadas nos Parquets:

- `adAccount_id`
- `adAccount_name`
- `campaign_id`
- `campaign_name`
- `context_timestamp`

## Como executar (local)

1. Instale as dependencias:

```bash
pip install -r requirements.txt
```

2. Coloque os arquivos parquet em `data/raw_campaigns/` ou aponte para uma pasta externa.

3. Modelos: por segurança, não versionar arquivos `.joblib`. Coloque seu modelo em
   uma pasta externa (por exemplo `/home/vroston/data/models/modelo_fadiga.joblib`) e
   defina a variável de ambiente `CAMPAIGN_MODEL_PATH` apontando para ele.

4. Rode a aplicacao:

```bash
CAMPAIGN_DATA_PATH=/home/vroston/data/raw_campaigns \
CAMPAIGN_MODEL_PATH=/home/vroston/data/models/modelo_fadiga.joblib \
streamlit run app/main.py
```

## Como executar com Docker

```bash
docker compose up --build
```

O compose foi configurado para montar `/home/vroston/data` em `/app/data` como somente leitura
e define automaticamente `CAMPAIGN_DATA_PATH=/app/data/raw_campaigns` e
`CAMPAIGN_MODEL_PATH=/app/data/models/modelo_fadiga.joblib` dentro do container.

## Fluxo do Dashboard

- Sidebar com filtro de `adAccount`.
- Ao selecionar a conta, o app lista apenas as campanhas dessa conta.
- Ao selecionar a campanha, o historico e carregado e ordenado por `context_timestamp`.
- Se o modelo estiver disponivel, a coluna `fadiga_prevista` e adicionada.
- Se nao houver modelo valido, o app continua funcionando e exibe uma metrica numerica.

## Observacoes de seguranca

- Processamento 100% local (sem APIs externas).
- `.gitignore` ignora `data/`, `*.parquet` e `*.joblib` para proteger dados e modelos.

Recomendações adicionais:

- Não adicione dados ou modelos ao repositório. Se você já cometeu esses arquivos,
  use `git rm --cached <path>` para removê-los do índice e depois faça um commit.
- Use a variável `CAMPAIGN_DATA_PATH` para apontar para a pasta centralizada
  (ex: `/home/vroston/data/raw_campaigns`). O código fará apenas leitura dessa pasta.
- Para criar um link simbólico local em vez de exportar a variável, execute:

```bash
./scripts/link_data.sh /home/vroston/data/raw_campaigns
```
