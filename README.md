# Prognóstico de Campanha — Modelo 72h

Dashboard Streamlit para prognóstico de campanhas de Meta Ads. O modelo classifica campanhas em **potencial** ou **risco** baseado no comportamento das primeiras 72 horas.

## Modelo

**Prognosis 72h v2** — Classificador binário LightGBM que prevê se uma campanha vai atingir o target de eficiência (CPA):

- **ROC-AUC holdout**: 0.9571
- **Precision**: 82.5% | **Recall**: 92.6%
- **26 features** calculadas nas primeiras 72h
- **Threshold calibrado**: 0.2178 (F2-score)

Artefatos em `/data/models/prognosis_72h/`:
- `lgbm_prognosis_72h.joblib` — modelo treinado
- `model_summary.json` — métricas e feature importance
- `dataset_72h.parquet` — dataset de treinamento

## Requisitos

- Python 3.11+
- Arquivos Parquet horários em `data/raw_campaigns/`
- Modelo e artefatos em `/data/models/prognosis_72h/`

## Setup

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

### 2. Configure o arquivo `.env`

Crie ou edite o arquivo `.env` na raiz do projeto:

```bash
# .env
export CAMPAIGN_DATA_PATH=/home/vroston/data/raw_campaigns
export CAMPAIGN_MODEL_ROOT=/home/vroston/data/models
```

(Já existe um `.env` com esses valores padrão.)

### 3. Rode a aplicação

```bash
streamlit run app/main.py
```

Ou carregue o `.env` primeiro:

```bash
source .env && streamlit run app/main.py
```

A aplicação abre em `http://localhost:8501` por padrão.

## Estrutura da Aplicação

### Página 1: Resultados do Modelo

Exibe a saúde geral do modelo:
- **ROC-AUC**: 0.9571
- **Precision/Recall**: cards com as métricas holdout
- **Feature Importance**: top 15 features (gráfico horizontal)
- **Cross-Validation**: tabela com os 5 folds (AUC, PR-AUC, iterações)
- **Matriz de Confusão**: 2×2 com TP/FP/FN/TN

### Página 2: Análise de Campanha

Análise preditiva por campanha:

1. **Seleção**: escolha uma campanha do dropdown
2. **Resultado**: card com cor (verde = potencial, vermelho = risco)
   - Probabilidade em destaque
   - 3 métricas rápidas (Gasto 72h, Spend vs Budget, CPC)
3. **Evolução**: gráfico de linha mostrando a probabilidade ao longo das 72h
4. **Detalhes**: tabela expansível com as 26 features por categoria

## Tratamento de Dados

### Features (26 no total)

**Volume:**
- `spend_72h`, `spend_daily_avg`, `nc_72h` (cliques), `nic_72h` (conversões)

**Eficiência:**
- `cpc_72h`, `cpa_72h`, `cc_72h` (tax de conv.), `nic_per_100_spend`

**Utilização:**
- `budget_daily`, `spend_vs_budget`, `burn_rate`

**Target:**
- `budget_vs_target`, `cpa_vs_target`, `cpc_vs_target`, `target_configured`

**Trajetória:**
- `cpc_trend_72h`, `spend_trend_72h`

**Configuração:**
- `active_ads`, `pct_ads_active`, `active_adsets`
- `is_cbo`, `is_sales`, `is_leads`, `is_engagement`, `is_ecommerce`, `is_infoprodutos`

### Filtragem

- Apenas snapshots com `campaign_status == 'ACTIVE'`
- Apenas primeiras 72h (`age_hours <= 72`)
- Mínimo 3 snapshots para análise confiável

### Tratamento de NaN

- Features ausentes são preenchidas com NaN (exibidas como "—")
- Divisões seguras usam clipping (máximo 10x)

## Segurança

- ✓ Processamento 100% local (sem APIs externas)
- ✓ Dados nunca são salvos (cache por sessão apenas)
- ✓ Modelos e dados em `.gitignore`
- ✓ Arquivo `.env` ignorado (caminhos e secrets)

Use a variável `CAMPAIGN_DATA_PATH` para apontar para a pasta **centralizada** de dados:

```bash
export CAMPAIGN_DATA_PATH=/home/vroston/data/raw_campaigns
```

## Troubleshooting

**Erro: "Model not found"**
- Verifique `CAMPAIGN_MODEL_ROOT` no `.env`
- Confirme que `/data/models/prognosis_72h/lgbm_prognosis_72h.joblib` existe

**Erro: "No campaigns found"**
- Verifique `CAMPAIGN_DATA_PATH`
- Certifique-se de que os `.parquet` existem em `data/raw_campaigns/`

**Análise lenta**
- A primeira carga de campanhas leva ~1 minuto (cache TTL: 1 hora)
- Snapshots são cacheados por campanha
- Use `streamlit cache clear` se desejar resetar

## Desenvolvimento

O código está estruturado em:
- `app/main.py` — entry point e navegação
- `app/pages/model_results.py` — página de métricas do modelo
- `app/pages/campaign_analysis.py` — página de análise de campanha
- `app/features.py` — cálculo das 26 features
- `app/config.py` — carregamento do modelo e configurações
