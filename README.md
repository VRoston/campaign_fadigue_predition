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

### Colocando artefatos do modelo

O app procura por artefatos de modelo de forma flexível. Você pode colocar os arquivos de modelo de duas maneiras:

- Diretório externo (recomendado): coloque os artefatos em `/home/victor/data/models/<campaign_id>/` com os nomes esperados:
  - `xgb_alert_model.joblib` (modelo joblib)
  - `label_encoders.joblib` (opcional)
  - `model_summary.json` (opcional, com `alert_threshold`, `metrics_holdout_auc`, etc.)
    Em seguida, defina `CAMPAIGN_MODEL_ROOT=/home/victor/data/models` e `CAMPAIGN_RESULT_PATH=/home/victor/data/result_campaigns`.

- Diretório embutido no repositório (conveniência para desenvolvimento): o app também detecta artefatos colocados em `app/models/`.
  Por exemplo, você pode adicionar `app/models/xgb_alert_model.joblib`, `app/models/label_encoders.joblib` e `app/models/model_summary.json`.

Exemplo de execução apontando para um diretório de modelos e destino de resultados:

```bash
CAMPAIGN_DATA_PATH=/home/victor/data/raw_campaigns \
CAMPAIGN_MODEL_ROOT=/home/victor/data/models \
CAMPAIGN_RESULT_PATH=/home/victor/data/result_campaigns \
streamlit run app/main.py
```

### Usando um arquivo `.env`

Para facilitar o run em máquinas diferentes, crie um arquivo `.env` na raiz do projeto com as variáveis necessárias (existe `.env.example` como modelo). Exemplos de conteúdo:

```bash
# .env (exemplo)
export CAMPAIGN_DATA_PATH=/home/victor/data/raw_campaigns
export CAMPAIGN_MODEL_ROOT=/home/victor/data/models
export CAMPAIGN_RESULT_PATH=/home/victor/data/result_campaigns
export ALERT_THRESHOLD=0.16821053624153137
```

Depois carregue as variáveis e execute:

```bash
source .env
streamlit run app/main.py
```

Nota: `.env` está incluído em `.gitignore` para evitar comitar caminhos/segredos locais. Use `.env.example` como referência.

Observação: se os artefatos estão em `app/models/` (dentro do repositório), não é necessário definir `CAMPAIGN_MODEL_ROOT`.

Dependências adicionais

- Para carregar o `xgboost` e os artefatos serializados pode ser necessário instalar `scikit-learn` e `xgboost`. Elas já foram adicionadas ao `requirements.txt`.

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

## Novas Páginas da UI

Adicionei duas melhorias na UI (Streamlit):

- **Dashboard (padrão)**: visão geral por campanha com KPIs, timeline (risco de fadiga), lista de alertas, análise de drivers, saúde do modelo e recomendações de ação.
- **Schema / Mapeamento de Colunas**: página que agrega um mapeamento das colunas presentes nos arquivos Parquet (amostra) mostrando dtypes e contagens não-nulas — útil para entender espaços em branco / NaNs esperados.

Alterne entre as páginas usando o seletor `Página` na barra lateral do Streamlit.

Observações sobre NaNs: os dados provenientes da API Meta podem conter muitos valores ausentes por design. A UI indica a presença de valores ausentes e não quebra — o modelo e as heurísticas tratam NaNs quando necessário.
