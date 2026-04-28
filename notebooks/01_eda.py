# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.3",
#     "pandas>=2.3.3",
#     "numpy>=1.24.0",
#     "matplotlib>=3.7.0",
#     "seaborn>=0.13.2",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    # 📊 Resumo Executivo
    Visão geral dos principais insights acionáveis descobertos durante a Análise Exploratória, focados em **Estratégia de Negócio** e **Modelagem Preditiva**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
        **🎯 O Desafio do Churn (Variável Alvo)**
        A taxa de cancelamento global da base é de **~26.5%**.
        Como a base é **desbalanceada** (73.5% vs 26.5%), nossos modelos de Machine Learning precisarão de técnicas de balanceamento (como pesos de classe ou SMOTE) e deverão ser avaliados por métricas sensíveis a minorias, como **F1-Score** ou **ROC-AUC**, abandonando a acurácia simples.
        """),
        kind="warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "💼 Alavancas de Negócio (Ações de Retenção)": mo.md(r"""
        - **Fidelização Contratual:** O contrato mensal (`Month-to-month`) tem um churn assustador de **42.7%**. Migrar clientes para contratos de 1 ano derruba a taxa para **11.3%**. Campanhas agressivas de *upgrade* para planos anuais são a maior alavanca financeira.
        - **Fricção de Pagamento:** Clientes que pagam por *Electronic check* (45.3% de churn) saem quase 3x mais que os de débito automático (~16%). Oferecer descontos para o cadastro em cartão de crédito/débito automático reduz o atrito e a inadimplência.
        - **O Paradoxo da Fibra Óptica:** O serviço de internet mais caro/rápido é o que mais perde clientes (41.9% de churn). É crucial o negócio investigar se o motivo é sensibilidade ao preço (`MonthlyCharges` alto) ou baixa qualidade/instabilidade técnica da fibra entregue.
        - **Lock-in com Serviços Extras:** Clientes sem `OnlineSecurity` ou `TechSupport` têm churn superior a 40%. Oferecer esses serviços de forma "gratuita" nos primeiros 3 meses cria uma forte barreira de saída técnica e psicológica.
        """),
            "🤖 Diretrizes para Modelagem (Feature Engineering)": mo.md(r"""
        - **Multicolinearidade Crítica:** `TotalCharges` possui correlação altíssima com `tenure` (o valor total é apenas tempo $\times$ mensalidade). Para modelos lineares (Regressão Logística), `TotalCharges` deve ser **removida** para evitar ruído matemático.
        - **Remoção de Features Irrelevantes:** A feature `gender` provou-se estatisticamente inútil para a decisão de churn (Feminino 27.0% vs Masculino 26.2%). Deve ser descartada para não gerar ruído no modelo.
        - **Foco no Início da Jornada:** A variável `tenure` tem fortíssima correlação negativa com churn (-0.35). O risco é máximo no início do ciclo de vida. O modelo de ML precisará ser extremamente preciso em classificar clientes novos (0 a 6 meses).
        - **Codificação de Variáveis:** Features binárias como `SeniorCitizen` devem ser tratadas corretamente (e não como variáveis contínuas numéricas) em algoritmos baseados em árvores e florestas (Random Forest, XGBoost).
        """),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo, plt):
    # Gráfico Executivo: O Perfil de Alto Risco
    fig_exec, ax_exec = plt.subplots(figsize=(6, 4))
    top_churners = {
        "Pagamento: Cheque Eletrônico": 45.3,
        "Contrato: Mês a Mês": 42.7,
        "Internet: Fibra Óptica": 41.9,
        "Sem Segurança Online": 41.8,
        "Perfil: Terceira Idade": 41.7,
    }
    # Ordenar para o gráfico de barras horizontais
    labels = list(top_churners.keys())[::-1]
    values = list(top_churners.values())[::-1]

    ax_exec.barh(labels, values, color="#d62728", alpha=0.85)
    ax_exec.set_xlim(0, 55)
    ax_exec.set_title(
        "Top 5 Fatores de Risco de Churn", fontsize=13, weight="bold", pad=15
    )
    ax_exec.set_xlabel("Taxa de Cancelamento (%)", fontsize=10)

    # Adicionar os rótulos de dados
    for i, v in enumerate(values):
        ax_exec.text(
            v + 1,
            i,
            f"{v}%",
            va="center",
            weight="bold",
            color="#d62728",
            fontsize=10,
        )

    # Limpar bordas
    ax_exec.spines["top"].set_visible(False)
    ax_exec.spines["right"].set_visible(False)
    ax_exec.spines["bottom"].set_visible(False)
    ax_exec.spines["left"].set_visible(False)
    ax_exec.tick_params(axis="y", length=0)
    plt.tight_layout()

    # Renderiza no marimo
    mo.vstack(
        [
            mo.md(
                "**Visão Executiva:** Grupos com taxa de evasão superior a 40% (Ação Imediata)"
            ),
            fig_exec,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.1 Mapa de Risco RFM: Tempo de Vida × Ticket Médio

    **Quanto mais vermelho/intenso, maior a taxa de cancelamento.**
    """)
    return


@app.cell(hide_code=True)
def _(df_eda, pd, plt, sns):
    # Heatmap RFM Executivo: Tenure vs MonthlyCharges
    fig_rfm_exec, ax_rfm_exec = plt.subplots(figsize=(8, 5))

    rfm_exec = pd.crosstab(
        df_eda["tenure_segment"],
        df_eda["monetary_segment"],
        values=df_eda["Churn_bin"],
        aggfunc="mean",
    ).mul(100)

    # Reorder rows for better narrative
    row_order = [
        "Novo (0-12m)",
        "Fiel (13-24m)",
        "Premium (25-48m)",
        "Veterano (49m+)",
    ]
    rfm_exec = rfm_exec.reindex([r for r in row_order if r in rfm_exec.index])

    if sns is not None:
        sns.heatmap(
            rfm_exec,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            cbar_kws={"label": "Churn %", "shrink": 0.8},
            linewidths=0.5,
            ax=ax_rfm_exec,
        )

    ax_rfm_exec.set_title(
        "⚠️ Taxa de Churn (%) por Segmento", fontsize=14, weight="bold", pad=15
    )
    ax_rfm_exec.set_xlabel("Ticket Médio (Mensalidade)", fontsize=11)
    ax_rfm_exec.set_ylabel("Tempo de Casa (Tenure)", fontsize=11)

    plt.tight_layout()
    fig_rfm_exec
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **📊 Insights do Mapa de Risco:**

    - **🔴 Zona Crítica (Novo + Premium):** Clientes com menos de 1 ano pagando mensalidades altas são o grupo de **maior risco absoluto** da base. Sem fidelidade contratual, saem assim que sentirem que o custo não compensa.
    - **🟡 Zona de Atenção (Novo + Padrão):** Mesmo pagando valores médios, a falta de tempo de casa mantém o churn elevado. Aqui cabem campanhas de desconto progressivo nos primeiros 6 meses.
    - **🟢 Zona Segura (Veterano + Qualquer ticket):** Após 4 anos, o cliente está "preso" à operadora. O churn despenca independente do valor pago. O foco deve ser upsell, não retenção.
    - **💡 Ação Imediata para Diretoria:** Priorizar 100% da verba de retenção nos primeiros 12 meses de vida do cliente. Após esse período, o ROI de campanhas de retenção cai drasticamente.
    """)
    return


@app.cell(hide_code=True)
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md("""
    # ⚙️ Processamento de Dados

    Importações, carregamento e tratamento da base.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.1 Configuração do ambiente
    """)
    return


@app.cell(hide_code=True)
def _():
    from __future__ import annotations

    import warnings

    import matplotlib

    matplotlib.use("Agg")  # Backend não-interativo (obrigatório para WASM)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore")

    try:
        import seaborn as sns

        sns.set_style("whitegrid")
    except Exception:
        sns = None

    plt.rcParams["figure.figsize"] = (12, 6)
    return mtick, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2 Carregamento e Processamento dos dados

    Caminho relativo à raiz do projeto:
    - `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2.1 Carrega Base
    """)
    return


@app.cell
def _(pd):
    # Lógica inteligente: tenta arquivo local primeiro (desenvolvimento),
    # depois URL pública (WASM/molab)
    local_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    public_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

    try:
        df = pd.read_csv(local_path)
        print(f"✓ Dataset carregado do arquivo local: {local_path}")
    except (FileNotFoundError, OSError):
        print(f"⚠ Arquivo local não encontrado. Carregando da URL pública...")
        df = pd.read_csv(public_url)
        print(f"✓ Dataset carregado da URL: {public_url}")

    print(f"Shape dos dados: {df.shape}")
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2.2.2 Copia de Data RAW
    """)
    return


@app.cell
def _(df):
    df_1 = df.copy()
    return (df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2.2.3 Total Charges para Numerico
    """)
    return


@app.cell(hide_code=True)
def total_charges_to_numeric_explain(mo):
    mo.md(r"""
    **Por que essa conversão é necessária?**
    Conforme observamos na **Seção 3.1 (Describe da Base)** da nossa coluna de Análises, o `TotalCharges` foi importado originalmente como texto (`object`). Isso ocorreu porque a base continha espaços em branco para clientes novos sem faturas. O `pd.to_numeric(errors="coerce")` força a conversão para número, transformando esses espaços ocultos em `NaN` para podermos tratá-los depois.
    """)
    return


@app.cell
def _(df_1, pd):
    df_2 = df_1.assign(
        TotalCharges=pd.to_numeric(df_1["TotalCharges"], errors="coerce")
    )
    return (df_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2.2.4 Tratamento de Nulos
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Remoção de Nulos (Drop NA):**
    Como visto em 3.2, os valores nulos representam uma fatia minúscula (0.16%) referente a clientes sem a 1ª fatura. A remoção direta simplifica a modelagem preditiva posterior e não introduz vieses significativos na base de dados.
    """)
    return


@app.cell
def dropna_tranform(df_2):
    df_3 = df_2.dropna()
    return (df_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2.5 Categoriza Senior Citzen
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Senioridade Categórica:**
    A feature `SeniorCitizen` foi importada originalmente como `0` e `1` (numérica). O mapeamento explícito para texto (`No`/`Yes`) previne que gráficos de correlação ou algoritmos de Machine Learning a tratem como uma variável contínua, facilitando também a compreensão visual, para o EDA.(Depois ela deve ser trasnformada novamente para modelagem)
    """)
    return


@app.cell
def seniorcitzen_transform(df_3):
    df_4 = df_3.assign(
        SeniorCitizen=df_3["SeniorCitizen"].map({0: "No", 1: "Yes"})
    )
    return (df_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2.2.6 Criação dos Atributos RFM
    """)
    return


@app.cell
def _(df_4):
    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "MultipleLines",
    ]

    df_5 = df_4.assign(
        rfm_recency=df_4["tenure"],
        rfm_frequency=(df_4[service_cols] == "Yes").sum(axis=1),
        rfm_monetary=df_4["MonthlyCharges"],
    )
    return (df_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2.2.7 Segmentação por Tempo de Vida (Recency)
    """)
    return


@app.cell
def _(df_5, pd):
    df_6 = df_5.assign(
        tenure_segment=pd.cut(
            df_5["rfm_recency"],
            bins=[0, 12, 24, 48, float("inf")],
            labels=[
                "Novo (0-12m)",
                "Fiel (13-24m)",
                "Premium (25-48m)",
                "Veterano (49m+)",
            ],
            right=True,
            include_lowest=True,
        )
    )
    return (df_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2.2.8 Segmentação por Ticket Médio (Monetary)
    """)
    return


@app.cell
def _(df_6, pd):
    df_7 = df_6.assign(
        monetary_segment=pd.qcut(
            df_6["rfm_monetary"],
            q=3,
            labels=["Econômico", "Padrão", "Premium"],
        )
    )
    return (df_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2.2.9 Segmentação por Serviços Contratados (Frequency)
    """)
    return


@app.cell
def _(df_7, pd):
    df_8 = df_7.assign(
        service_segment=pd.cut(
            df_7["rfm_frequency"],
            bins=[-1, 0, 2, 4, 7],
            labels=[
                "Básico (0 serviços)",
                "Light (1-2)",
                "Plus (3-4)",
                "Full (5-7)",
            ],
        )
    )
    return (df_8,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2.2.10 Criação da Target Binária
    """)
    return


@app.cell
def _(df_8):
    df_eda = df_8.assign(Churn_bin=(df_8["Churn"] == "Yes").astype(int))
    return (df_eda,)


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md("""
    # 3. Análise Exploratória de Dados
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.1 Describe da Base
    """)
    return


@app.cell(hide_code=True)
def describe_insight(mo):
    mo.md(r"""
    - **Tipagem Incorreta:** A coluna `TotalCharges` foi importada como `object` (texto) em vez de numérica. Isso é um forte indício de que existem valores ocultos (como espaços em branco) inseridos no lugar de números.
    - **Falsa Numérica:** A variável `SeniorCitizen` aparece no `describe()` com valores de 0 a 1, mas trata-se de uma categoria binária (Sim/Não) e não de uma medida contínua.
    - **Variação de Planos:** O valor mensal cobrado (`MonthlyCharges`) possui uma amplitude grande, variando de \$18.25 a \$118.75, indicando perfis de clientes bem distintos (desde planos básicos até pacotes premium).
    - **Tempo de Retenção (`tenure`):** Temos desde clientes recém-chegados (0 meses) até clientes que estão na operadora há 6 anos (72 meses).
    """)
    return


@app.cell
def _(df):
    print("=== INFORMAÇÕES GERAIS ===\n")
    print(df.info())

    print("\n=== ESTATÍSTICAS DESCRITIVAS (NUMÉRICAS) ===\n")
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.2 Detectando Missings
    """)
    return


@app.cell
def missing_analysis(df_2, pd):
    print("=== ANÁLISE DE MISSING VALUES ===")
    missing_count = df_2.isnull().sum()
    missing_count = missing_count[missing_count > 0]

    if not missing_count.empty:
        missing_pct = (missing_count / len(df_2) * 100).round(2)
        missing_df = pd.DataFrame(
            {"Valores Nulos": missing_count, "Percentual (%)": missing_pct}
        ).sort_values(by="Percentual (%)", ascending=False)

        print(missing_df)
    else:
        print("Nenhum missing value detectado na base!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Como tratar esses dados nulos?**

    Os valores nulos identificados acima na coluna `TotalCharges` representam apenas **~0.16%** de toda a base de dados (11 clientes num total de 7043). Como o volume é ínfimo e não causa perda significativa de informação preditiva (além de representar clientes novos sem fatura gerada), a abordagem mais simples e robusta é a **remoção dessas linhas** (Drop NA).

    *Nota: Este tratamento já foi implementado de forma reativa na nossa pipeline de Processamento de Dados (Etapa 5 - Coluna Central), garantindo que as análises visuais abaixo utilizem a base `df_eda` 100% limpa.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.3 Análise da variável target (`Churn`)
    """)
    return


@app.cell
def _(df, df_3):
    print("=== AVALIAÇÃO DA COLUNA ORIGINAL (Churn) ===")
    print("A coluna Churn vem com os valores:")
    print(df_3["Churn"].value_counts(dropna=False))
    print(f"\nTotal de nulos na coluna original: {df['Churn'].isnull().sum()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Atributo Churn é uma string**, iremos transforma-lá em 0,1 para facilitar a análise.
    """)
    return


@app.cell
def distr_target(df_eda, plt):
    target_plot = plot_target_distribution(df_eda, "Churn", plt)
    target_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Variável Target:** A variável alvo `Churn` está desbalanceada (ratio ~0.36), o que reforça a necessidade de avaliar métricas além de acurácia.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.4 Análise das features categóricas
    """)
    return


@app.cell(hide_code=True)
def distr_feat_cat(df_eda, mo):
    cat_cols = (
        df_eda.select_dtypes(include="object")
        .columns.drop(["customerID", "Churn"], errors="ignore")
        .tolist()
    )
    cat_dropdown = mo.ui.dropdown(
        options=cat_cols,
        value=cat_cols[0],
        label="🎯 Selecione a Feature Categórica:",
    )
    cat_dropdown
    return (cat_dropdown,)


@app.cell
def _(cat_dropdown, df_eda, mtick, plt):
    plot_categorial = plot_single_categorical(
        df_eda, cat_dropdown.value, plt, mtick
    )
    plot_categorial
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Principais Insights das Features Categóricas:**

    Ao navegar pelo menu interativo acima, podemos extrair conclusões essenciais sobre o perfil da nossa base de clientes:

    - **👥 Demografia & Relacionamento:** A distribuição de gênero e parceiros é altamente simétrica (~50/50). No entanto, **70.2%** dos clientes não possuem dependentes (`Dependents=No`) e **83.8%** não são da terceira idade (`SeniorCitizen=No`).
    - **🌐 Serviços Base:** Quase todos possuem serviço telefônico (`PhoneService=Yes` em 90.3%). No mercado de internet, a **Fibra Óptica** é o carro-chefe com 44% de penetração, seguida pelo DSL (34.4%). Cerca de 21.6% não possuem internet.
    - **🛡️ Serviços Adicionais (Add-ons):** A adoção de serviços protetivos como `OnlineSecurity` e `TechSupport` é baixa (~29%). A grande maioria dos clientes prefere pacotes sem essas seguranças extras.
    - **⚠️ Vínculo e Risco Contratual (Atenção):** A característica que mais chama atenção é o vínculo financeiro. **55.1%** dos clientes possuem contratos `Month-to-month` (mensais), o que facilita muito a taxa de abandono (Churn). Além disso, o método de pagamento mais comum é o `Electronic check` (33.6%), sugerindo que o atrito mensal de faturamento é alto comparado a métodos automatizados.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.5 Análise de Outliers
    """)
    return


@app.cell(hide_code=True)
def outlier_analysis(df_eda, mo):
    num_cols = (
        df_eda.select_dtypes(include=["number"])
        .columns.drop(["Churn_bin"], errors="ignore")
        .tolist()
    )
    num_dropdown = mo.ui.dropdown(
        options=num_cols,
        value=num_cols[0],
        label="📏 Selecione a Feature Numérica (Outliers):",
    )
    num_dropdown
    return (num_dropdown,)


@app.cell
def _(df_eda, np, num_dropdown, pd, plt, sns):
    plot_outlier = plot_single_outlier(
        df_eda, num_dropdown.value, plt, sns, np, pd
    )
    plot_outlier
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Outliers:** Nenhum outlier crítico detectado nas variáveis numéricas.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.6 Análise de anomalias e duplicados
    """)
    return


@app.cell(hide_code=True)
def _(df_eda):
    print("=== ANOMALIAS E CONSISTÊNCIA ===\n")
    anomalies = []
    tenure_invalid = df_eda[(df_eda["tenure"] < 0) | (df_eda["tenure"] > 120)]
    if len(tenure_invalid) > 0:
        anomalies.append(f"tenure fora de 0-120: {len(tenure_invalid)}")
    monthly_invalid = df_eda[df_eda["MonthlyCharges"] < 0]
    if len(monthly_invalid) > 0:
        anomalies.append(f"MonthlyCharges negativo: {len(monthly_invalid)}")
    total_invalid = df_eda[df_eda["TotalCharges"] < 0]
    if len(total_invalid) > 0:
        anomalies.append(f"TotalCharges negativo: {len(total_invalid)}")
    lower_total = df_eda[
        (df_eda["tenure"] > 1)
        & df_eda["TotalCharges"].notna()
        & (df_eda["TotalCharges"] < df_eda["MonthlyCharges"])
    ]
    if len(lower_total) > 0:
        anomalies.append(
            f"TotalCharges menor que MonthlyCharges para tenure>1: {len(lower_total)}"
        )
    yes_no_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn",
    ]
    for _col in yes_no_cols:
        invalid = df_eda[
            ~df_eda[_col].isin(["Yes", "No"]) & df_eda[_col].notna()
        ]
        if len(invalid) > 0:
            anomalies.append(
                f"{_col} com valor fora de Yes/No: {len(invalid)}"
            )
    if anomalies:
        print("Anomalias encontradas:")
        for _item in anomalies:
            print(f"- {_item}")
    else:
        print(
            "✓ Nenhuma anomalia crítica detectada nas validações de domínio."
        )
    duplicates = int(df_eda.duplicated().sum())
    print(f"\nRegistros duplicados: {duplicates}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Anomalias e Consistência:** Nenhuma anomalia de domínio detectada.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.7 Análise de distribuições
    """)
    return


@app.cell(hide_code=True)
def assimetria_analysis(df_eda, np, pd, plt):
    numeric_cols = (
        df_eda.select_dtypes(include=["number"])
        .columns.drop(["Churn_bin"], errors="ignore")
        .tolist()
    )
    print("=== ASSIMETRIA (SKEWNESS) E CURTOSE ===\n")
    distribution_stats = []
    for _col in numeric_cols:
        skewness = float(df_eda[_col].skew())
        kurtosis = float(df_eda[_col].kurtosis())
        distribution_stats.append(
            {
                "Coluna": _col,
                "Skewness": round(skewness, 3),
                "Kurtosis": round(kurtosis, 3),
                "Interpretação": "Normal"
                if abs(skewness) < 0.5
                else "Assimétrica à direita"
                if skewness > 0
                else "Assimétrica à esquerda",
            }
        )
    print(pd.DataFrame(distribution_stats))

    _n = len(numeric_cols)
    _ncols = 3
    _nrows = int(np.ceil(_n / _ncols))
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(14, 4 * _nrows))
    _axes = np.array(_axes).reshape(-1)
    for _idx, _col in enumerate(numeric_cols):
        _axes[_idx].hist(
            df_eda[_col].dropna(), bins=30, color="skyblue", edgecolor="black"
        )
        _axes[_idx].set_title(f"{_col} | Skew: {df_eda[_col].skew():.2f}")
        _axes[_idx].set_xlabel("Valor")
        _axes[_idx].set_ylabel("Frequência")
    for _ax in _axes[_n:]:
        _fig.delaxes(_ax)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Distribuições:** `tenure` apresenta distribuição **bimodal** (muitos clientes novos com tenure baixo e clientes fidelizados com tenure alto), refletindo dois perfis distintos de comportamento. `MonthlyCharges` tem leve assimetria à direita. `TotalCharges` é fortemente assimétrico à direita.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.8 Análise de correlações (data readiness)
    """)
    return


@app.cell
def correlation_analysis(df_eda, pd, plt, sns):
    plot_correlation_analysis(df_eda, plt, sns, pd)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insights da Matriz de Correlação:**

    - **Risco e Custos (Correlações Positivas):** Os contratos mensais (`Contract_Month-to-month`) despontam isoladamente como o **maior preditor positivo de Churn** (correlação de **0.40**). Logo atrás, a falta de segurança online e suporte técnico (`OnlineSecurity_No` com 0.34 e `TechSupport_No` com 0.33) e internet de fibra óptica (0.30) completam o topo do ranking de evasão.
    - **Fidelidade (Correlações Negativas):** Tempo de permanência (`tenure` com -0.35) e **Contratos de 2 anos** (`Contract_Two year` com -0.30) são os maiores blindadores contra o churn. Curiosamente, a ausência de serviço de internet (`No internet service` com -0.22) também tem forte correlação negativa, sugerindo que clientes apenas de telefonia são mais estáveis.
    - **Atenção (Multicolinearidade):** Como esperado, o `TotalCharges` tem uma altíssima correlação com `tenure` (o total é o tempo vezes a mensalidade). Além disso, a `InternetService_Fiber optic` tem forte relação direta com o aumento da `MonthlyCharges`. Para a modelagem (especialmente algoritmos lineares), o ideal é remover o `TotalCharges` para não confundir o modelo com informações redundantes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.9 Análise de Churn vs Features
    """)
    return


@app.cell
def _(df_eda, mo, np):
    numeric_cols_kde = (
        df_eda.select_dtypes(include=[np.number])
        .columns.drop(["Churn_bin"], errors="ignore")
        .tolist()
    )
    num_kde_dropdown = mo.ui.dropdown(
        options=numeric_cols_kde,
        value=numeric_cols_kde[0] if numeric_cols_kde else None,
        label="📊 Selecione a Variável Numérica (KDE):",
    )
    num_kde_dropdown
    return (num_kde_dropdown,)


@app.cell
def _(df_eda, num_kde_dropdown, plot_single_numeric_kde, plt, sns):
    plot_single_numeric_kde(df_eda, num_kde_dropdown.value, plt, sns)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Variáveis Numéricas vs Churn (KDE):** As curvas de densidade mostram claramente como os perfis se separam:
    - **Menor `tenure`**: a curva de churn (vermelho) está concentrada à esquerda, próxima de zero. Clientes saem cedo, antes de criar vínculo.
    - **Maior `MonthlyCharges`**: a curva de churn (vermelho) está deslocada para a direita, indicando que quem paga mais tende a cancelar mais.
    - **Menor `TotalCharges`**: consequência direta do menor tenure — a curva de churn acumula valores baixos, pois saem antes de acumular cobrança.
    """)
    return


@app.cell
def _(df_eda, mo):
    churn_cat_cols = (
        df_eda.select_dtypes(include="object")
        .columns.drop(["customerID", "Churn"], errors="ignore")
        .tolist()
    )
    churn_cat_dropdown = mo.ui.dropdown(
        options=churn_cat_cols,
        value="Contract"
        if "Contract" in churn_cat_cols
        else churn_cat_cols[0],
        label="📊 Churn por Feature Categórica:",
    )
    churn_cat_dropdown
    return (churn_cat_dropdown,)


@app.cell
def _(churn_cat_dropdown, df_eda, mtick, pd, plt):
    plot_churn_cat = plot_churn_by_categorical(
        df_eda, churn_cat_dropdown.value, plt, mtick, pd
    )
    plot_churn_cat
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insights Direcionáveis: Churn vs. Atributos Categóricos**

    Executando a análise cruzada para todos os atributos em relação ao cancelamento (`Churn = Yes`), os principais ofensores ficam claros:

    - 💸 **Pagamento e Contrato (Os Maiores Ofensores):**
      - **PaymentMethod:** Clientes que pagam via **Electronic check** possuem a maior taxa de churn absoluta da base (**45.3%**), mais que o dobro de qualquer outro método (como débito automático ou cartão, que ficam na casa dos 15-16%).
      - **Contract:** A flexibilidade do **Month-to-month** custa caro: **42.7%** de churn. Prender o cliente em um contrato de 1 ano despenca isso para 11.3%, e de 2 anos para residuais 2.8%.

    - 🌐 **Serviço de Internet e Add-ons:**
      - **InternetService:** Paradoxalmente, o serviço mais rápido (**Fiber optic**) tem o churn mais alto (**41.9%**), enquanto o DSL retém muito mais (19.0%). Clientes que só assinam telefone praticamente não cancelam (7.4%).
      - **Proteção e Suporte:** Não possuir serviços extras de segurança é um gatilho. Quem tem `OnlineSecurity=No` (41.8%) ou `TechSupport=No` (41.6%) cancela massivamente, ao passo que a adoção desses serviços baixa a evasão para a casa dos 14-15%.

    - 👨‍👩‍👧‍👦 **Perfil Demográfico:**
      - **SeniorCitizen:** Idosos cancelam numa proporção bem maior (**41.7%**) que os mais jovens (23.7%).
      - **Família:** Clientes solteiros (`Partner=No`: 33.0%) e sem dependentes (31.3%) evadem mais. Planos familiares parecem "engessar" a saída.
      - **Gênero:** Irrelevante para o modelo (Feminino 27.0% vs Masculino 26.2%).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.10 Segmentação RFM Implícita (Análise por Tempo de Vida)

    Dividindo os clientes em grupos baseados no tempo de permanência (`tenure`), conseguimos enxergar claramente o perfil de risco de cada ciclo de vida.
    """)
    return


@app.cell(hide_code=True)
def _(df_eda, pd, plt):
    # Gráfico de Churn por Segmento de Tenure (RFM - Recency)
    fig_seg, ax_seg = plt.subplots(figsize=(10, 6))

    seg_churn = (
        pd.crosstab(
            df_eda["tenure_segment"], df_eda["Churn"], normalize="index"
        )
        .mul(100)
        .reindex(columns=["No", "Yes"])
    )
    colors_seg = ["#1F77B4", "#D62728"]
    seg_churn.plot(
        kind="barh",
        stacked=True,
        color=colors_seg,
        ax=ax_seg,
        edgecolor="white",
        linewidth=1.2,
        rot=0,
    )

    ax_seg.set_title(
        "Taxa de Churn por Segmento de Tempo de Vida (RFM - Recency)",
        fontsize=15,
        weight="bold",
        pad=12,
    )
    ax_seg.set_xlabel("% Clientes", fontsize=12)
    ax_seg.set_ylabel("")
    ax_seg.set_xlim(0, 100)
    ax_seg.legend(title="Churn", loc="lower right", frameon=True)
    ax_seg.grid(axis="x", linestyle="--", alpha=0.35)
    ax_seg.spines["top"].set_visible(False)
    ax_seg.spines["right"].set_visible(False)

    for container in ax_seg.containers:
        ax_seg.bar_label(
            container,
            fmt="%.1f%%",
            label_type="center",
            color="white",
            weight="bold",
            fontsize=10,
        )

    plt.tight_layout()
    fig_seg
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insights da Segmentação RFM (Recency, Frequency, Monetary):**

    A análise RFM segmenta os clientes em 3 dimensões críticas para retenção:

    - **⏰ Recency (Tempo de Vida - `tenure`):**
      - **Novo (0-12m):** É o segmento mais crítico. O risco de churn é massivo nos primeiros meses. Campanhas de onboarding e primeiros 90 dias são essenciais.
      - **Fiel (13-24m):** Clientes que passaram do primeiro ano demonstram maior resiliência. A taxa de churn despenca drasticamente. Foco em upsell (venda de add-ons).
      - **Premium (25-48m):** Clientes consolidados com alta retenção. Candidatos ideais para programas de fidelidade e contratos de longo prazo (2 anos).
      - **Veterano (49m+):** O grupo mais fiel da base. Risco quase zero, mas devem ser mantidos engajados para evitar churn por saturação.

    - **💰 Monetary (Ticket Médio - `MonthlyCharges`):**
      - Clientes do segmento **Premium** (mensalidade alta) tendem a ser mais sensíveis ao preço, especialmente se usam Fibra Óptica. O risco de churn está diretamente correlacionado com o valor pago quando não há contrato de fidelidade.

    - **📡 Frequency (Serviços Contratados):**
      - Clientes com mais serviços adicionais (`OnlineSecurity`, `TechSupport`, etc.) criam maior "lock-in" técnico e psicológico. O gráfico de análise de serviços mostra que quem tem 0-1 serviços extras tem churn muito superior a quem tem 3+ serviços.

    **Para Modelagem:** As features `tenure_segment`, `monetary_segment` e `service_segment` são variáveis categóricas ordinais extremamente poderosas para os algoritmos de ML, capturando de forma clara o estágio do ciclo de vida, poder aquisitivo e engajamento do cliente.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.11 Análise Avançada RFM (Recency × Frequency × Monetary)

    Agora que temos os 3 segmentos RFM criados, vamos cruzá-los para identificar os micro-perfis de maior risco.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    rfm_segments = ["tenure_segment", "monetary_segment", "service_segment"]
    rfm_dropdown = mo.ui.dropdown(
        options=rfm_segments,
        value="monetary_segment",
        label="💰 Selecione o Segmento RFM:",
    )
    rfm_dropdown
    return (rfm_dropdown,)


@app.cell
def _(df_eda, mtick, pd, plot_churn_by_rfm_segment, plt, rfm_dropdown):
    plot_rfm = plot_churn_by_rfm_segment(
        df_eda, rfm_dropdown.value, plt, mtick, pd
    )
    plot_rfm
    return


@app.cell
def _(df_eda, pd, plot_rfm_heatmap, plt, sns):
    plot_rfm_heatmap(df_eda, plt, sns, pd)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Insights Avançados da Segmentação RFM:**

    - **💰 Monetary (Ticket Médio):** Clientes do segmento **Premium** (maiores mensalidades) combinados com contratos mensais representam o maior risco financeiro. A alta mensalidade sem fidelização é uma receita para churn.
    - **📡 Frequency (Serviços):** Clientes do segmento **Básico** (0 serviços extras) têm churn massivo (~45%), enquanto o segmento **Full** (5-7 serviços) retém muito melhor (~15%). Isso prova que cada serviço adicional contratado cria uma "barreira de saída" psicológica e técnica.
    - **⏰ × 💰 Cruzamento (Heatmap):** O mapa de calor acima revela que a combinação mais perigosa é **Novo (0-12m) + Premium (alto ticket)**. Esses clientes chegam pagando caro, não têm fidelidade ainda, e saem rapidamente se sentirem que o custo-benefício não bate. Ações: onboarding premium com gerente dedicado nos primeiros 90 dias.
    - **🎯 Ação de Retenção Imediata:** O modelo de ML deve priorizar a classificação do segmento **Novo + Econômico/Básico**, pois são os mais numerosos e voláteis. Oferecer desconto progressivo nos primeiros 6 meses pode segurar essa base.
    """)
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md("""
    # 4. 🛠️ Funções Auxiliares

    Funções para abstrair código de visualização e utilitários da base.
    """)
    return


@app.function(hide_code=True)
def plot_target_distribution(df, target_col, plt):
    print(f"=== DISTRIBUIÇÃO DA VARIÁVEL TARGET ({target_col}) ===\n")
    target_counts = df[target_col].value_counts()
    target_percentages = (
        df[target_col].value_counts(normalize=True) * 100
    ).round(2)
    print("Contagem:")
    print(target_counts)
    print("\nPercentual:")
    print(target_percentages)
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    _axes[0].bar(
        target_counts.index, target_counts.values, color=["lightblue", "coral"]
    )
    _axes[0].set_ylabel("Frequência")
    _axes[0].set_title(f"Distribuição da variável {target_col}")
    _axes[0].grid(axis="y", alpha=0.3)
    _axes[1].pie(
        target_counts.values,
        labels=target_counts.index,
        autopct="%1.1f%%",
        colors=["lightblue", "coral"],
        startangle=90,
    )
    _axes[1].set_title("Proporção de classes")
    plt.tight_layout()
    ratio = target_counts.min() / target_counts.max()
    print(f"\nRatio de balanceamento: {ratio:.2f}")
    if ratio < 0.5:
        print(
            "⚠️ Dataset desbalanceado: considerar class_weight/SMOTE em modelagem."
        )
    else:
        print("✓ Dataset razoavelmente balanceado.")
    return _fig


@app.function(hide_code=True)
def plot_single_categorical(df, col_name, plt, mtick):
    if not col_name:
        return
    plt.figure(figsize=(8, 5))
    pct = df[col_name].value_counts(normalize=True).mul(100)
    _ax = pct.plot(
        kind="bar",
        rot=0,
        color=["brown", "green", "coral", "lightblue", "purple"],
    )
    _ax.set_title(f"Distribuição: {col_name}")
    _ax.set_xlabel("")
    _ax.set_ylabel("% Clientes")
    _ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    _ax.grid(axis="y", alpha=0.3)
    for _p in _ax.patches:
        height = _p.get_height()
        _ax.text(
            _p.get_x() + _p.get_width() / 2,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    return plt.gcf()


@app.function(hide_code=True)
def plot_single_outlier(df, col_name, plt, sns, np, pd):
    if not col_name:
        return
    plt.figure(figsize=(10, 4))
    series = df[col_name].dropna()

    if sns is not None:
        ax = sns.boxplot(x=series, color="coral")
    else:
        plt.boxplot(series, vert=False, patch_artist=True)
        ax = plt.gca()

    ax.set_title(f"Análise de Outliers: {col_name}")
    ax.grid(axis="x", alpha=0.3)

    if len(series) > 1 and float(series.std(ddof=0)) > 0:
        z_scores = np.abs((series - series.mean()) / series.std(ddof=0))
        outlier_count = int((z_scores > 3).sum())
        outlier_pct = (outlier_count / len(series)) * 100
    else:
        outlier_count = 0
        outlier_pct = 0.0

    # Text box with stats
    textstr = f"Total de Outliers (Z > 3): {outlier_count}\nPercentual: {outlier_pct:.2f}%"
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    ax.text(
        0.95,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )

    plt.tight_layout()
    return plt.gcf()


@app.function(hide_code=True)
def plot_correlation_analysis(df, plt, sns, pd):
    df_corr = df.drop(columns=["customerID", "Churn"], errors="ignore").copy()

    # Para EDA visual, drop_first=False permite ver a correlação de TODAS as categorias
    # (ex: Contract_Month-to-month não desaparece)
    df_corr = pd.get_dummies(df_corr, drop_first=False)
    corr = df_corr.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))
    if sns is not None:
        sns.heatmap(corr, cmap="coolwarm", center=0, cbar_kws={"shrink": 0.8})
    else:
        plt.imshow(corr, cmap="coolwarm", aspect="auto")
        plt.colorbar(shrink=0.8)
    plt.title("Matriz de Correlação — Telco Churn")
    plt.tight_layout()
    fig = plt.gcf()

    print("=== TOP 10 CORRELAÇÕES POSITIVAS COM CHURN_BIN ===")
    target_corr = corr["Churn_bin"].sort_values(ascending=False)
    print(target_corr.head(11)[1:])  # skip Churn_bin itself
    print("\n=== TOP 10 CORRELAÇÕES NEGATIVAS COM CHURN_BIN ===")
    print(target_corr.tail(10))

    print(
        "\n=== MULTICOLINEARIDADE (Features independentes com |corr| > 0.7) ==="
    )
    high_pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = float(corr.iloc[i, j])
            if (
                abs(value) > 0.7
                and cols[i] != "Churn_bin"
                and cols[j] != "Churn_bin"
            ):
                high_pairs.append((cols[i], cols[j], round(value, 3)))

    if high_pairs:
        multicol_df = pd.DataFrame(
            high_pairs, columns=["Feature 1", "Feature 2", "Correlação"]
        )
        print(
            multicol_df.sort_values("Correlação", ascending=False).to_string(
                index=False
            )
        )
    else:
        print("Nenhuma correlação forte detectada.")
    return fig


@app.function(hide_code=True)
def plot_churn_by_categorical(df, feature_col, plt, mtick, pd):
    if not feature_col:
        return
    colors = ["#1F77B4", "#D62728"]  # No, Yes
    cross_tab = (
        pd.crosstab(df[feature_col], df["Churn"], normalize="index")
        .mul(100)
        .reindex(columns=["No", "Yes"])
    )

    _ax = cross_tab.plot(
        kind="bar",
        stacked=True,
        color=colors,
        rot=0,
        figsize=(9, 5.5),
        edgecolor="white",
        linewidth=1.2,
    )
    _ax.set_title(
        f"Análise de Churn por {feature_col}",
        fontsize=15,
        weight="bold",
        pad=12,
    )
    _ax.set_xlabel(feature_col, fontsize=12)
    _ax.set_ylabel("% Clientes", fontsize=12)
    _ax.set_ylim(0, 100)
    _ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    _ax.legend(title="Churn", loc="upper right", frameon=True)
    _ax.grid(axis="y", linestyle="--", alpha=0.35)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    for _p in _ax.patches:
        _h = _p.get_height()
        if _h >= 4:
            _x = _p.get_x() + _p.get_width() / 2
            _y = _p.get_y() + _h / 2
            _ax.text(
                _x,
                _y,
                f"{_h:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                weight="bold",
            )

    plt.tight_layout()
    return plt.gcf()


@app.cell(hide_code=True)
def _():
    def plot_churn_by_rfm_segment(df, segment_col, plt, mtick, pd):
        if not segment_col:
            return
        colors = ["#1F77B4", "#D62728"]
        cross_tab = (
            pd.crosstab(df[segment_col], df["Churn"], normalize="index")
            .mul(100)
            .reindex(columns=["No", "Yes"])
        )

        _ax = cross_tab.plot(
            kind="barh",
            stacked=True,
            color=colors,
            rot=0,
            figsize=(9, 5.5),
            edgecolor="white",
            linewidth=1.2,
        )
        _ax.set_title(
            f"Churn por Segmento RFM: {segment_col}",
            fontsize=15,
            weight="bold",
            pad=12,
        )
        _ax.set_xlabel("% Clientes", fontsize=12)
        _ax.set_ylabel("")
        _ax.set_xlim(0, 100)
        _ax.legend(title="Churn", loc="lower right", frameon=True)
        _ax.grid(axis="x", linestyle="--", alpha=0.35)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

        for container in _ax.containers:
            _ax.bar_label(
                container,
                fmt="%.1f%%",
                label_type="center",
                color="white",
                weight="bold",
                fontsize=10,
            )

        plt.tight_layout()
        return plt.gcf()

    def plot_rfm_heatmap(df, plt, sns, pd):
        rfm_crosstab = pd.crosstab(
            df["tenure_segment"],
            df["monetary_segment"],
            values=df["Churn_bin"],
            aggfunc="mean",
        ).mul(100)

        plt.figure(figsize=(10, 6))
        if sns is not None:
            sns.heatmap(
                rfm_crosstab,
                annot=True,
                fmt=".1f",
                cmap="YlOrRd",
                cbar_kws={"label": "Taxa de Churn (%)"},
            )
        else:
            plt.imshow(rfm_crosstab, cmap="YlOrRd", aspect="auto")
            plt.colorbar(label="Taxa de Churn (%)")
        plt.title(
            "Mapa de Calor RFM: Churn (%) por Tenure vs Ticket Médio",
            fontsize=14,
            weight="bold",
            pad=15,
        )
        plt.xlabel("Segmento Monetário (MonthlyCharges)", fontsize=12)
        plt.ylabel("Segmento de Tempo de Vida (Tenure)", fontsize=12)
        plt.tight_layout()
        return plt.gcf()

    def plot_single_numeric_kde(df, col_name, plt, sns):
        if not col_name or sns is None:
            return
        plt.figure(figsize=(10, 5))
        sns.kdeplot(
            data=df,
            x=col_name,
            hue="Churn",
            fill=True,
            common_norm=False,
            palette={"No": "#1F77B4", "Yes": "#D62728"},
            alpha=0.5,
        )
        plt.title(
            f"Distribuição de {col_name} por Churn",
            fontsize=14,
            weight="bold",
            pad=12,
        )
        plt.xlabel(col_name, fontsize=11)
        plt.ylabel("Densidade", fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        return plt.gcf()

    return plot_churn_by_rfm_segment, plot_rfm_heatmap, plot_single_numeric_kde


@app.cell(hide_code=True)
def _():
    return


if __name__ == "__main__":
    app.run()
