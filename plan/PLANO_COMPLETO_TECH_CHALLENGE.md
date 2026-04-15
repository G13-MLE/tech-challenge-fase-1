# Plano Completo - Tech Challenge Fase 1

## Guia Passo a Passo para o Time

**Projeto:** Pipeline end-to-end de ML para previsão de churn em telecomunicações  
**Time:** Grupo 13 (5 membros)  
**Tech Leads:** Eduardo (DevOps) e Fernando (ML)  
**Dataset:** Telco Customer Churn (IBM)

### Resumo do Time

| Membro | Papel | Fortalezas (nível 3) | Área Principal |
|--------|-------|---------------------|----------------|
| **Eduardo** | Tech Lead DevOps | Git(3), Docker(3), Cloud(3), Documentação(3) | Infra, Deploy, Qualidade |
| **Fernando** | Tech Lead ML | EDA(3), Jupyter(3), Métricas ML(3), Pandera(3) | Modelagem, Arquitetura ML |
| **Rafael** | Team Member | Jupyter(3), Conceitos DL(3), Cloud(3) | Deep Learning, Deploy |
| **Bruno** | Team Member | Pandas(3), EDA(2), scikit-learn(2) | Dados, ML Clássico |
| **Ygor** | Team Member | EDA(2), ML Canvas(2) | Desenvolvimento, Negócio |

---

## 1. Levantamento de Skills da Equipe

### 1.1 Template de Pesquisa

**Formato:** Google Forms ou Planilha compartilhada

**Campos obrigatórios:**
- Nome do membro
- Email
- Timestamp
- 21 habilidades com escala 0-3:
  - **0** = Sem experiência
  - **1** = Iniciante (já vi/teoria)
  - **2** = Intermediário (já usei em projetos)
  - **3** = Avançado (posso ensinar outros)

**Lista completa de skills:**

1. Git/GitHub e estrutura de projeto
2. Python project tooling (pyproject.toml, ruff, Makefile)
3. Jupyter Notebooks
4. Docker
5. Pandas / manipulação de dados
6. EDA (análise exploratória)
7. Entendimento de negócio / ML Canvas
8. scikit-learn (baselines, pipelines)
9. Métricas ML (AUC-ROC, PR-AUC, F1, FP/FN)
10. PyTorch (construção de MLP)
11. Conceitos: arquitetura, loss, training loop, early stopping
12. MLflow (experiment tracking)
13. pytest (unit, schema, smoke tests)
14. Pandera (schema validation)
15. Refatoração em módulos Python (src/)
16. FastAPI (endpoints /predict e /health)
17. Pydantic (validação de dados)
18. Logging estruturado
19. Documentação de arquitetura de deploy
20. Vídeo no formato STAR
21. Cloud deploy (AWS/Azure/GCP) — opcional

### 1.2 Processo de Coleta

1. **Criar formulário** (Google Forms)
2. **Enviar para todos** via Discord/email
3. **Prazo:** 24-48 horas
4. **Lembrete:** A cada 12h no Discord
5. **Consolidar** em planilha CSV

### 1.3 Análise das Respostas

**Checklist de validação:**
- [x] Todos os 5 membros responderam? ✅ (Eduardo, Fernando, Rafael, Bruno, Ygor)
- [x] Nenhum campo em branco? ✅
- [x] Valores entre 0-3 apenas? ✅
- [x] Nomes padronizados? ✅

**Identificação de gaps:**
- Calcular média por skill
- Destacar skills com média < 1.0 (crítico)
- Identificar mentores (skill ≥ 2)

---

## 2. Distribuição de Tarefas

### 2.1 Definição de Tech Leads e Time

**Time completo:** 5 membros (2 Tech Leads + 3 Team Members)

#### Tech Leads

| Tech Lead | Área de Domínio | Skills Chave | Responsabilidade Principal |
|-----------|-----------------|--------------|---------------------------|
| **Eduardo** | DevOps/Qualidade | Git(3), Docker(3), Cloud(3), Doc(3) | Infraestrutura, deploy, pytest, documentação |
| **Fernando** | ML/Modelagem | EDA(3), Métricas(3), Pandera(3) | Arquitetura PyTorch, modelagem, code review |

#### Team Members

| Membro | Skills Destaque | Função nas Sprints | Pair Programming |
|--------|----------------|-------------------|------------------|
| **Rafael** | DL Concepts(3), Jupyter(3), Cloud(3), PyTorch(2) | Co-lidera DL com Fernando, deploy com Eduardo | Mentor DL/Cloud |
| **Bruno** | Pandas(3), scikit-learn(2), PyTorch(1) | ML/Dados, apoio EDA | Aprendiz → Mentor em dados |
| **Ygor** | EDA(2), ML Canvas(2) | Negócio, documentação | Aprendiz em tooling/ML |

#### Matriz de Skills do Time (Consolidado)

| Skill | Eduardo | Fernando | Rafael | Bruno | Ygor | Média | Status |
|-------|:-------:|:--------:|:------:|:-----:|:----:|:-----:|:------:|
| Git/GitHub | 3 | 1 | 1 | 1 | 1 | 1.4 | 🟡 |
| Python Tooling | 1 | 1 | 2 | 0 | 0 | 0.8 | 🔴 |
| Jupyter | 1 | 3 | 3 | 2 | 1 | 2.0 | 🟢 |
| Docker | 3 | 1 | 2 | 1 | 0 | 1.4 | 🟡 |
| Pandas | 1 | 2 | 2 | 3 | 1 | 1.8 | 🟢 |
| EDA | 0 | 3 | 1 | 2 | 2 | 1.6 | 🟡 |
| ML Canvas | 0 | 2 | 2 | 1 | 2 | 1.4 | 🟡 |
| scikit-learn | 1 | 2 | 2 | 2 | 0 | 1.4 | 🟡 |
| Métricas ML | 0 | 3 | 2 | 1 | 0 | 1.2 | 🟡 |
| **PyTorch** | 0 | 1 | **2** | 1 | 0 | **0.8** | 🟡 |
| **DL Concepts** | 0 | 2 | **3** | 1 | 0 | **1.2** | 🟡 |
| MLflow | 0 | 2 | 2 | 1 | 0 | 1.0 | 🟡 |
| pytest | 0 | 2 | 2 | 0 | 0 | 0.8 | 🔴 |
| Pandera | 0 | 3 | 1 | 0 | 0 | 0.8 | 🔴 |
| Refatoração | 2 | 2 | 1 | 0 | 0 | 1.0 | 🟡 |
| **FastAPI** | 2 | 2 | **2** | 0 | 0 | **1.2** | 🟡 |
| Pydantic | 0 | 0 | 1 | 0 | 0 | 0.2 | 🔴 |
| Logging | 2 | 1 | 2 | 0 | 0 | 1.0 | 🟡 |
| Model Card | 0 | 1 | 2 | 1 | 1 | 1.0 | 🟡 |
| Doc Deploy | 3 | 1 | 2 | 1 | 1 | 1.6 | 🟡 |
| Vídeo STAR | 0 | 2 | 1 | 1 | 1 | 1.0 | 🟡 |
| **Cloud Deploy** | 3 | 0 | **3** | 1 | 1 | **1.6** | 🟢 |

**Legenda:** 🟢 Média ≥ 2.0 | 🟡 Média 1.0-1.9 | 🔴 Média < 1.0 (crítico)

### 2.2 Regras de Pair Programming

**Quando usar:**
- Skill gap < 1.0 para mais de 2 membros
- Tecnologia crítica para o projeto (PyTorch, FastAPI, Cloud)
- Primeira vez do membro com a tecnologia
- **Novo:** Co-mentoria quando há 2+ experts (Fernando + Rafael em DL)

**Estrutura do par:**
- **Mentor:** Skill ≥ 2 na área
- **Aprendiz:** Skill 0-1
- **Co-mentoria:** Dois mentores (ex: Fernando + Rafael em PyTorch)
- **Duração:** Mínimo 1-2h por sessão
- **Ferramenta:** VS Code Live Share, Discord screen share, GitHub Codespaces

### 2.3 Template de Alocação

```
SPRINT [X]: [NOME]
├── Responsável Principal: [Nome]
├── Apoio: [Nome] (se houver)
├── Pair Programming:
│   ├── Mentor: [Nome]
│   └── Aprendiz: [Nome]
├── Entregáveis:
│   ├── [Item 1]
│   ├── [Item 2]
│   └── [Item 3]
└── Critérios de Aceitação:
    ├── [Critério 1]
    └── [Critério 2]
```

---

## 3. Execução das Sprints

### Sprint 0: Setup (Dias 1-2)

**Objetivo:** Preparar ambiente de desenvolvimento

**Entregáveis:**

1. **Estrutura de pastas:**
   ```
   projeto/
   ├── src/
   ├── data/
   ├── models/
   ├── tests/
   ├── notebooks/
   ├── docs/
   └── plan/
   ```

2. **Arquivos de configuração:**
   - `pyproject.toml` (dependências, ruff, pytest)
   - `.gitignore` (Python, ML, IDE)
   - `Makefile` (targets: install, lint, test, run)
   - `README.md` inicial

3. **Dataset baixado e validado**

**Checklist:**
- [ ] Python 3.9+ instalado
- [ ] Virtual environment criado
- [ ] Dependências instaladas (`make install`)
- [ ] Git repo inicializado
- [ ] Primeiro commit realizado
- [ ] Dataset no diretório `data/`

**Rituais:**
- Standup: Definir quem faz o quê
- Review: Validar estrutura juntos

---

### Sprint 1: EDA & ML Canvas (Dias 3-5)

**Objetivo:** Entender dados e definir estratégia de negócio

**Entregáveis:**

1. **Notebook EDA** (`notebooks/01_eda.ipynb`):
   - Análise de volume e qualidade
   - Distribuição de features
   - Tratamento de missing values
   - Visualizações

2. **ML Canvas preenchido:**
   - Stakeholders identificados
   - Problema de negócio definido
   - Métricas de sucesso
   - SLOs estabelecidos

3. **Baselines treinados:**
   - DummyClassifier (estratificado)
   - LogisticRegression
   - Métricas calculadas: AUC-ROC, PR-AUC, F1
   - Registrados no MLflow

**Checklist:**
- [ ] EDA executa sem erros
- [ ] Todos os gráficos salvos
- [ ] ML Canvas revisado pelo time
- [ ] Baselines com métricas comparáveis
- [ ] Experimentos visíveis no MLflow UI

**Rituais:**
- Daily: 15min no Discord (o que fiz, o que vou fazer, bloqueios)
- Review: Apresentar EDA para o time

---

### Sprint 2: Modelagem com MLP (Dias 6-10)

**Objetivo:** Construir e treinar rede neural com força-tarefa de DL

**Distribuição de Tarefas:**

| Sub-tarefa | Responsáveis | Mentor Técnico |
|------------|--------------|----------------|
| Arquitetura MLP | Fernando + Rafael | Co-mentoria (ambos nível 2+) |
| Training loop | Rafael (lead) + Bruno (apoio) | Fernando (review) |
| Early stopping | Fernando (implementa) | - |
| Comparação modelos | Bruno + Ygor | Fernando |
| MLflow tracking | Rafael + Eduardo | - |

**Entregáveis:**

1. **Modelo MLP em PyTorch:**
   - Arquitetura definida (camadas, ativações)
   - Loss function apropriada
   - Training loop com validação
   - Early stopping implementado
   - **Novo:** Documentação das decisões de arquitetura

2. **Comparação de modelos:**
   - Tabela comparativa (baselines + MLP)
   - Análise de trade-off FP vs FN
   - Gráficos de learning curves
   - **Novo:** Justificativa da arquitetura escolhida

3. **Artefatos no MLflow:**
   - Parâmetros do modelo
   - Métricas de treino/validação/teste
   - Modelo serializado (.pt ou .pkl)
   - Requirements usados
   - **Novo:** Tags indicando responsáveis

**Checklist:**
- [ ] MLP treina sem erros
- [ ] Early stopping funciona
- [ ] Métricas >= baselines
- [ ] Overfitting controlado
- [ ] Artefatos versionados no MLflow
- [ ] Documentação técnica atualizada
- [ ] Todos os 5 membros participaram de pelo menos 1 pair session

**Rituais:**
- Daily: Foco em resolver bugs de treinamento
- **Pair sessions estruturadas:**
  - Sessão 1: Fernando + Rafael (arquitetura avançada)
  - Sessão 2: Rafael + Bruno (implementação)
  - Sessão 3: Fernando + Ygor (conceitos DL)
- Review: Demo do modelo + explicação das escolhas técnicas

---

### Sprint 3: API & Testes (Dias 11-14)

**Objetivo:** Produzir código de qualidade com API funcional

**Distribuição de Tarefas:**

| Componente | Responsável Principal | Suporte | Review |
|------------|----------------------|---------|---------|
| Estrutura src/ | Rafael | Eduardo | Fernando |
| Pipeline sklearn | Bruno | Fernando | - |
| API FastAPI | Eduardo (lead) + Rafael | Ygor | Fernando |
| Testes pytest | Rafael + Fernando | Bruno | Eduardo |
| Pandera schemas | Fernando (lead) | Rafael | - |
| Logging | Eduardo + Rafael | - | - |

**Entregáveis:**

1. **Código refatorado:**
   - Estrutura `src/` organizada
   - Módulos separados (data, model, api, utils)
   - Pipelines sklearn reprodutíveis
   - **Novo:** Documentação de cada módulo

2. **API FastAPI:**
   - Endpoint POST `/predict` (input JSON, output predição)
   - Endpoint GET `/health` (status da API)
   - Validação Pydantic dos inputs
   - Logging estruturado (JSON)
   - Middleware de latência
   - **Novo:** Documentação automática (OpenAPI/Swagger)

3. **Testes automatizados** (mínimo 5, aumentado de 3):
   - Testes unitários (funções críticas) - Rafael
   - Testes de schema (Pandera) - Fernando
   - Smoke test (API sobe e responde) - Ygor
   - **Novo:** Testes de integração - Eduardo
   - **Novo:** Testes de performance - Bruno

**Checklist:**
- [ ] `make test` passa todos os testes (5+)
- [ ] `make lint` não reporta erros críticos
- [ ] API responde em < 200ms (local)
- [ ] Logs estruturados funcionando
- [ ] Pipeline reproduz resultado da sprint 2
- [ ] Documentação OpenAPI acessível em `/docs`
- [ ] Testes de cobertura > 70%

**Rituais:**
- Daily: Status dos testes e coverage
- **Pair sessions:**
  - Sessão 1: Eduardo + Rafael (arquitetura API)
  - Sessão 2: Fernando + Bruno (Pandera schemas)
  - Sessão 3: Rafael + Ygor (pytest básico)
- Review: Testar API localmente + revisar coverage

---

### Sprint 4: Documentação & Entrega (Dias 15-17)

**Objetivo:** Documentar e preparar entrega final com força-tarefa

**Distribuição de Tarefas:**

| Entregável | Responsável Principal | Revisores | Apoio |
|------------|----------------------|-----------|-------|
| Model Card | Fernando | Bruno, Ygor | Rafael (contribuições técnicas) |
| Doc Arquitetura | Eduardo (lead) + Rafael | Fernando | Bruno |
| README | Ygor | Todos | - |
| Vídeo STAR | **Todos participam** (partes individuais) | Tech Leads | - |
| Deploy Cloud | Eduardo + Rafael | - | Bruno, Ygor (testes) |

**Entregáveis:**

1. **Model Card** (`docs/model_card.md`):
   - Performance em dados de teste
   - Limitações identificadas
   - Vieses e fairness
   - Cenários de falha
   - **Novo:** Contribuições de cada membro

2. **Documentação de arquitetura** (`docs/architecture.md`):
   - Diagrama de fluxo (batch vs real-time)
   - Plano de monitoramento
   - Métricas e alertas
   - Playbook de incidentes
   - **Novo:** Decisões técnicas justificadas

3. **README final:**
   - Descrição do projeto
   - Instruções de setup
   - Como rodar localmente
   - Como fazer deploy
   - **Novo:** Créditos do time

4. **Vídeo STAR** (5 minutos) - **Estrutura por membro:**
   - **Situation:** Ygor (30s) - Contexto do problema
   - **Task:** Bruno (45s) - O que precisava ser feito
   - **Action:** Rafael (2min) - Arquitetura e implementação técnica
   - **Result:** Eduardo (1min) - Deploy e resultados
   - **Encerramento:** Fernando (45s) - Aprendizados e próximos passos
   - **Transições:** Gravação em conjunto ou edição

5. **Deploy cloud (Agora Obrigatório com 2 experts):**
   - API acessível publicamente
   - URL documentada
   - **Novo:** Testes de carga básicos
   - **Novo:** Documentação de troubleshooting
   - Responsáveis: Eduardo (3) + Rafael (3)

**Checklist:**
- [ ] Toda documentação revisada por pelo menos 2 pessoas
- [ ] README claro e completo
- [ ] Vídeo STAR gravado, editado e revisado
- [ ] Todos os artefatos no GitHub
- [ ] **Deploy cloud funcional e testado**
- [ ] URL pública documentada
- [ ] Testes de carga realizados
- [ ] Retrospectiva do time documentada

**Rituais:**
- Daily: Revisão de documentos
- **Sessão de gravação do vídeo:** 1h em grupo
- Review geral: Todo time valida entrega
- **Deploy day:** Eduardo + Rafael (2h focadas)
- Retrospectiva: Lições aprendidas + celebração

---

## 4. Checkpoints e Validações

### Checkpoints por Sprint

**Antes de iniciar cada sprint:**
- [ ] Sprint anterior foi aceita?
- [ ] Issues criadas no GitHub?
- [ ] Responsáveis definidos?
- [ ] Pares de programação agendados?

**Durante a sprint:**
- [ ] Daily realizada?
- [ ] Bloqueios identificados e resolvidos?
- [ ] Commits frequentes (mínimo 1/dia)?

**Ao final da sprint:**
- [ ] Todos os entregáveis prontos?
- [ ] Checklist de qualidade passou?
- [ ] Code review realizado?
- [ ] Documentação atualizada?

### Critérios de Aceitação por Etapa

**Setup:**
- Repo clonável e rodável em qualquer máquina
- `make install` funciona
- Estrutura de pastas padronizada

**EDA:**
- Notebook executável
- Insights claros sobre os dados
- Decisões de pré-processamento documentadas

**Modelagem:**
- Modelo salvo e versionado
- Métricas reprodutíveis
- Overfitting não crítico

**API:**
- Testes passando
- API documentada (OpenAPI/Swagger)
- Logging funcional

**Documentação:**
- Model Card completo
- Instruções claras
- Vídeo com qualidade adequada

### Quando Pedir Ajuda aos Tech Leads

**Problemas técnicos:**
- Erro que não resolve em 30 min
- Dúvida arquitetural
- Conflito de merge complexo

**Problemas de gestão:**
- Membro não responde em 24h
- Entrega em risco
- Necessidade de redistribuir tarefas

**Comunicar via:**
- Discord (canal #tech-challenge ou DM)
- Marcar nos comentários da issue
- Tag `@tech-lead` no GitHub

---

## 5. FAQ (Perguntas Frequentes)

### Sobre o Projeto

**Q: Qual é o objetivo do Tech Challenge?**  
A: Construir um pipeline end-to-end de ML para previsão de churn em telecomunicações, desde a análise de dados até a API de predição deployada.

**Q: Quais são as entregas obrigatórias?**  
A: Repositório GitHub com código funcional, modelos treinados, API, testes, documentação e vídeo STAR. Deploy em cloud é bônus.

**Q: Qual o dataset usado?**  
A: Telco Customer Churn da IBM (disponível no Kaggle).

### Sobre Skills

**Q: Não tenho experiência em PyTorch. Vou conseguir?**  
A: Sim! Agora temos **Fernando + Rafael** como mentores de PyTorch (ambos nível 2+). O pair programming nas sprints 2 e 3 vai te ajudar, e você terá suporte de dois experts.

**Q: Como funciona o nívelamento de skills?**  
A: Membros com mais experiência (skill ≥ 2) fazem pair programming com quem está começando (skill 0-1). Todos aprendem e entregam juntos.

**Q: O que acontece se eu não preencher a pesquisa de skills?**  
A: O planejamento fica incompleto e pode afetar a distribuição de tarefas. Preencha assim que possível!

### Sobre Pair Programming

**Q: Como são formados os pares?**  
A: Baseado na pesquisa de skills: mentor (skill ≥ 2) + aprendiz (skill 0-1). Os tech leads supervisionam.

**Q: Quanto tempo dura uma sessão de pair?**  
A: Mínimo 1 hora, mas pode ser mais se necessário. Agende com antecedência.

**Q: Posso fazer pair programming sozinho se souber a tecnologia?**  
A: Sim, mas é recomendado fazer code review com outro membro mesmo assim.

### Sobre GitHub e Issues

**Q: Quando criar uma issue?**  
A: Para cada tarefa significativa: bugs, novas features, dúvidas técnicas, documentação. Fique a vontade para criar issues, elas irão passar pelo processo de grooming antes de serem de fato aceitas para Ready.

**Q: Como nomear as issues?**  
A: Use prefixos: `[EDA]`, `[MODEL]`, `[API]`, `[TEST]`, `[DOC]`, `[BUG]`. Não é obrigatório mas ajuda, pode usar também as labels do GitHub

**Q: Quando fazer commit?**  
A: Faça commits pequenos e frequentes (mínimo 1 por dia de trabalho). Mensagens claras: "feat: adiciona baseline logistic regression".

### Sobre Rituais

**Q: Quais são os rituais obrigatórios?**  
A: Daily (15min), Sprint Planning (início), Sprint Review (final), Retrospectiva (após sprint 4). Pode ser feito assíncrono também.

**Q: Onde acontecem as reuniões?**  
A: No Discord do grupo. Agende com antecedência. Ou assíncrono, para assuntos menos complexos, ou pontuais.

**Q: O que levar para o Status Report?**  
A: O que você fez desde o último report, o que vai fazer até o próximo, e se tem algum bloqueio. É interessante reportar com antecedência máxima, caso esteja bloqueado em algo.

### Sobre Entregas

**Q: E se eu não conseguir entregar no prazo?**  
A: Comunique imediatamente no Discord. Os tech leads podem redistribuir ou ajustar escopo. Não deixe atrasar, ou ficar bloqueado por muito tempo.

**Q: Posso entregar antes do prazo?**  
A: Sim! Quanto antes melhor, assim sobra tempo para revisão e ajustes.

**Q: Como saber se minha entrega está boa?**  
A: Use os checklists de cada sprint. Se passou em todos os itens, está ótimo! e se tem aval do lead da tarefa.

### Sobre Tecnologias

**Q: Posso usar outras bibliotecas além das especificadas?**  
A: Pode, mas mantenha as obrigatórias: PyTorch, FastAPI, scikit-learn, MLflow, pytest, Pandera.

**Q: Preciso usar Docker?**  
A: Altamente recomendado para garantir reprodutibilidade. Eduardo pode ajudar.

**Q: E se eu tiver problema com alguma instalação?**  
A: Pergunte no canal #tech-challenge no Discord. Alguém do time vai ajudar.

### Sobre Avaliação

**Q: Como o projeto é avaliado?**  
A: Por critérios com pesos: Código (20%), Rede Neural (25%), Pipeline (15%), API (15%), Documentação (10%), Vídeo STAR (10%), Deploy (5% bônus).

**Q: O que é o vídeo STAR?**  
A: Apresentação de 5 minutos no formato: Situação, Tarefa, Ação, Resultado. Grave mostrando tela e explicando o projeto.

**Q: O deploy em cloud é obrigatório?**  
A: Não, mas vale 5% da nota. Se fizer, use AWS, Azure ou GCP.

---

## 6. Referências e Recursos

### Documentação Oficial

- **PyTorch:** https://pytorch.org/docs/
- **FastAPI:** https://fastapi.tiangolo.com/
- **scikit-learn:** https://scikit-learn.org/stable/
- **MLflow:** https://mlflow.org/docs/latest/index.html
- **pytest:** https://docs.pytest.org/
- **Pandera:** https://pandera.readthedocs.io/

### Templates Úteis

- **ML Canvas:** https://www.madewithml.com/courses/mlops/product-design/
- **Model Card:** https://modelcards.withgoogle.com/about
- **Cookiecutter Data Science:** https://drivendata.github.io/cookiecutter-data-science/

### Comunidades

- Discord ML Ops Brasil
- PyTorch Forums
- FastAPI Discord

---

## 7. Contatos e Responsáveis

### Estrutura do Time

| Papel | Responsável | Discord/GitHub | Skills Chave | Disponibilidade |
|-------|-------------|----------------|--------------|-----------------|
| **Tech Lead DevOps** | Eduardo | @eduardonunesp | Git(3), Docker(3), Cloud(3) | Alto |
| **Tech Lead ML** | Fernando | @fernando | EDA(3), Métricas(3), Pandera(3) | Alto |
| **Team Member** | **Rafael** | @gabipasse | **DL(3), Jupyter(3), Cloud(3)** | Recuperando acesso Discord |
| **Team Member** | Bruno | @bruno | Pandas(3), scikit-learn(2) | Médio |
| **Team Member** | Ygor | @ygor | EDA(2), ML Canvas(2) | Médio |

**Canal Discord:** `#tech-challenge`  
**Comunicação alternativa:** Issues do GitHub (todos têm acesso)

### Pares de Programação por Sprint

| Sprint | Par 1 | Par 2 | Par 3 |
|--------|-------|-------|-------|
| **0 (Setup)** | Eduardo → Ygor (Git/tooling) | Rafael → Bruno (Docker/Python) | - |
| **1 (EDA)** | Fernando → Bruno (EDA/ML) | Rafael → Ygor (Jupyter/Canvas) | - |
| **2 (MLP)** | Fernando ↔ Rafael (co-mentoria DL) | Rafael → Bruno (PyTorch) | Fernando → Ygor (conceitos) |
| **3 (API)** | Eduardo ↔ Rafael (FastAPI/arch) | Fernando → Bruno (Pandera) | Rafael → Ygor (pytest) |
| **4 (Deploy)** | Eduardo + Rafael (Cloud) | Bruno + Ygor (Testes/Doc) | Todos (Vídeo STAR) |

### Responsáveis por Área Crítica

| Área | Expert 1 | Expert 2 | Suporte |
|------|----------|----------|---------|
| **PyTorch/DL** | Fernando (1) | **Rafael (3)** | Bruno (1) |
| **FastAPI** | Fernando (2) | **Rafael (2)** | Eduardo (2) |
| **Cloud Deploy** | **Eduardo (3)** | **Rafael (3)** | - |
| **EDA** | **Fernando (3)** | Bruno (2) | Rafael (1) |
| **Testes** | Fernando (2) | **Rafael (2)** | Eduardo (0) |
| **Git/DevOps** | **Eduardo (3)** | Rafael (1) | Ygor (1) |

---

## Próximos Passos Imediatos

### ✅ Completados
- [x] **Pesquisa de skills:** Todos os 5 membros responderam (incluindo Rafael @gabipasse)
- [x] **Consolidação de dados:** Matriz de skills atualizada com Rafael
- [x] **Análise de gaps:** Identificados e mapeados (ver seção 2.1)

### 🎯 Próximas Ações (Prioridade Alta)
1. [ ] **Onboarding do Rafael:**
   - Adicionar ao Discord do time (novo link enviado)
   - Apresentar estrutura do projeto
   - Sincronizar com pair programming da Sprint 2
   
2. [ ] **Criar issues para Sprint 0:**
   - Issue #1: Setup do repositório (Eduardo + Ygor)
   - Issue #2: Configuração Docker/Python (Rafael + Bruno)
   - Issue #3: Dataset e validação inicial (Fernando)
   
3. [ ] **Agendar reunião de kickoff:**
   - Apresentação do plano atualizado
   - Definição de horários para pair programming
   - Alinhamento de expectativas
   
4. [ ] **Preparação técnica:**
   - Criar template de repositório
   - Configurar MLflow local
   - Validar acesso ao dataset

### 📊 Análise Pós-Rafael

**Melhorias identificadas:**
- ✅ PyTorch: De 0.8 (4 membros) para 0.8 (5 membros) - Rafael nível 2 compensa distribuição
- ✅ Cloud Deploy: De 1.6 para 1.6 - Rafael nível 3 fortalece par com Eduardo
- ✅ FastAPI: De 1.2 para 1.2 - Rafael nível 2 adiciona terceiro expert
- ✅ Jupyter: De 2.0 para 2.2 - Rafael nível 3 reforça

**Gaps que persistem (precisam de atenção):**
- 🔴 Pydantic: 0.2 (apenas Rafael nível 1)
- 🔴 Python Tooling: 0.8 (Ygor 0, Bruno 0)
- 🔴 pytest: 0.8 (Ygor 0, Bruno 0)
- 🟡 Git: 1.4 (apenas Eduardo nível 3)

**Recomendação:** Usar Rafael como "ponte" entre áreas - ele tem skills intermediárias/avançadas em múltiplas áreas críticas.

---

*Documento criado em: Março 2026*  
*Última atualização: 26 de Março de 2026 (inclusão do Rafael e atualização do time para 5 membros)*
