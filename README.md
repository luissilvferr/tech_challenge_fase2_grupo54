# Tech Challenge — Fase 2 | POSTECH Data Analytics

> **INSTRUÇÕES:** este README é um template. Substitua **todos** os blocos marcados com
> `<!-- PREENCHER -->` e apague as linhas de instrução antes de submeter.

---

## 1. Identificação

| Campo | Valor |
|---|---|
| Turma | 2DTATBB |
| Grupo | Grupo 54 |
| Data de entrega | 07/10/2026|

### Integrantes

| Nome completo | RM | E-mail |
|---|---|---|
| LUIS ENRIQUE CRUZ SILVA FERREIRA | RM377788 | rm377788@fiap.com.br|
---

## 2. Links da entrega

Estes três links são **obrigatórios** e devem ser idênticos aos do PDF de submissão.

| Item | Link |
|---|---|
| Repositório | https://github.com/luissilvferr/tech_challenge_fase2_grupo54/|
| Vídeo executivo (≤ 5 min) | https://docs.google.com/videos/d/1jLkND3QPQ6_YL2PUOTeS25m2UVxrUqGba3xZe14AeZU/play?usp=sharing |
| Apresentação | https://drive.google.com/file/d/1Ac1WnachxKIqwsohYy3xYa92v-EZSeI8/view?usp=sharing|

> ⚠️ Repositório privado ou inacessível inviabiliza a avaliação da entrega.
> Confira o acesso em uma janela anônima antes de enviar.

---

## 3. O problema
A instituição precisa decidir diariamente quem deve ter o cartão de crédito aprovado, separando os "bons" dos "maus" pagadores com base no perfil pessoal e financeiro de cada solicitante. 
A motivação para utilizar Machine Learning está na possibilidade de analisar simultaneamente diversas características dos clientes, como renda, idade, ocupação, situação familiar e histórico profissional, identificando padrões associados ao comportamento de pagamento. Dessa forma, o modelo pode apoiar a instituição na identificação de clientes com maior ou menor risco, tornando a análise mais ágil e consistente.

### Variável alvo
A variável-alvo (`ALVO`) é binária: vale **1** quando o cliente apresentou atraso de **60 dias ou mais** em qualquer mês do histórico e **0** caso contrário. Ela não existe nos dados brutos e foi construída a partir da coluna `STATUS` do `credit_record`, que registra, para cada cliente e cada mês, se havia empréstimo (`X`), se o crédito foi quitado (`C`) ou qual era a faixa de atraso (`0` a `5`, variando de menos de 30 dias até 150 dias ou mais). Para cada cliente, foi considerado o **pior status observado no histórico** (`WORST_STATUS`), aplicando-se posteriormente o limiar `WORST_STATUS >= 2`.

**Por que binarizar?** `WORST_STATUS` possui 8 categorias, sendo que as faixas de atraso mais graves são raras: os status 3 a 5 representam apenas **0,83% dos clientes (302)**, enquanto os status 2 a 5 correspondem a **1,69% (616)**. Prever cada nível separadamente exigiria a utilização de um modelo multiclasse, com poucas observações em algumas categorias, o que poderia gerar estimativas instáveis. Além disso, a decisão de negócio é essencialmente binária: **identificar ou não um cliente como de maior risco de inadimplência**.

Optou-se por seguir o padrão consolidado na análise de risco de crédito e adotar o limiar de **60 dias ou mais de atraso** para caracterizar o cliente como "mau pagador". Embora a utilização de um limiar de 30 dias ou mais resultasse em um conjunto de dados menos desbalanceado, essa abordagem poderia classificar como maus pagadores clientes que apresentaram atrasos pontuais ou operacionais, mas que não necessariamente possuem um perfil de alto risco de crédito.

### Dataset

| Campo | Valor |
|---|---|
| Fonte | https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction |
| Linhas × colunas | application_record.csv: 438557 x 18; credit_record.csv: 1048575 x 3 |
| Período / versão | <!-- PREENCHER --> |
| Licença de uso | <!-- PREENCHER --> |

Descrição das variáveis:
* **application_record**

| Variável | Tipo | Descrição |
| --- | --- | --- |
| ID | int64 | Número do cliente |
| CODE_GENDER | str | Gênero |
| FLAG_OWN_CAR | str | Possui carro |
| FLAG_OWN_REALTY | str | Possui imóvel |
| CNT_CHILDREN | int64 | Número de filhos |
| AMT_INCOME_TOTAL | float64 | Renda anual |
| NAME_INCOME_TYPE | str | Categoria de renda |
| NAME_EDUCATION_TYPE | str | Nível de escolaridade |
| NAME_FAMILY_STATUS | str | Estado civil |
| NAME_HOUSING_TYPE | str | Tipo de moradia |
| DAYS_BIRTH | int64 | Data de nascimento (contagem regressiva a partir do dia atual (0), -1 significa ontem) |
| DAYS_EMPLOYED | int64 | Data de início do emprego (contagem regressiva a partir do dia atual (0). Se for positivo, significa que a pessoa está desempregada atualmente) |
| FLAG_MOBIL | int64 | Possui celular |
| FLAG_WORK_PHONE | int64 | Possui telefone do trabalho |
| FLAG_PHONE | int64 | Possui telefone residencial |
| FLAG_EMAIL | int64 | Possui e-mail |
| OCCUPATION_TYPE | str | Ocupação / Profissão |
| CNT_FAM_MEMBERS | float64 | Tamanho da família |

* **credit_record**

| Variável         | Tipo    | Descrição                                                                                                                                                                                                                                                                                                                                                |
| ---------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ID             | int64 | Identificador único do cliente.                                                                                                                                                                                                                                                                                                                          |
| MONTHS_BALANCE | int64 | Mês de referência do registro, expresso de forma regressiva. **0** representa o mês mais recente (mês atual na base), **-1** o mês anterior, **-2** o mês retrasado, e assim sucessivamente.                                                                                                                                                             |
| STATUS         | str   | Status do crédito no mês de referência. <br> **0:** 1 a 29 dias de atraso. <br> **1:** 30 a 59 dias de atraso. <br> **2:** 60 a 89 dias de atraso. <br> **3:** 90 a 119 dias de atraso. <br> **4:** 120 a 149 dias de atraso. <br> **5:** atraso superior a 150 dias. <br> **C:** crédito quitado no mês. <br> **X:** sem registro de empréstimo no mês. |

## 4. Como reproduzir

```bash
git clone https://github.com/luissilvferr/tech_challenge_fase2_grupo54
cd tech_challenge_fase2_grupo54

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

Baixe o dataset e coloque o arquivo bruto em `data/raw/` (os dados **não** são versionados —
veja `data/README.md`).

Depois execute os notebooks nesta ordem:

| # | Notebook | O que faz |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | Análise exploratória |
| 2 | `notebooks/02_preprocessamento.ipynb` | Limpeza, escala e feature engineering |
| 3 | `notebooks/03_modelagem.ipynb` | Treino e comparação dos modelos |
| 4 | `notebooks/04_avaliacao.ipynb` | Métricas, importância de variáveis e conclusões |

**Semente fixa:** `RANDOM_STATE = 42`, declarada na primeira célula de cada notebook.
Rodar os notebooks na ordem acima, a partir de um ambiente limpo, deve reproduzir
exatamente os números da seção 5.

---

## 5. Resultados

| Modelo | Acurácia | Precisão | Recall | F1 | AUC-ROC |
| --- | --- | --- | --- | --- | --- |
| XGBoost | 0.984 | 0.747 | 0.047 | 0.088 | 0.621 |
| Random Forest | 0.977 | 0.150 | 0.066 | 0.089 | 0.589 |
| Logistic Regression | 0.602 | 0.019 | 0.446 | 0.036 | 0.549 |
| Dummy Baseline | 0.983 | 0.000 | 0.000 | 0.000 | 0.500 |

**Modelo escolhido:** O XGBoost foi escolhido por ser o modelo que melhor ordena os clientes por risco, que é o que importa em uma base com apenas 1,7% de maus pagadores. A acurácia não distingue os modelos (o Dummy, que aprova todos, tem 0,983 e o XGBoost 0,984), e o recall e o F1 no corte padrão de 0,50 dependem mais de como cada modelo escala suas probabilidades do que da qualidade do ranking: a regressão logística tem o maior recall (0,446), mas só porque sinaliza muita gente, e sua precisão (0,019) é quase igual à taxa de maus pagadores, com o pior AUC-ROC entre os modelos treinados (0,549). O XGBoost teve o melhor AUC-ROC (0,621, contra 0,589 da Random Forest e 0,500 do acaso) e a melhor PR-AUC (0,106, contra 0,041 da Random Forest), as duas métricas que avaliam a ordenação dos clientes sem depender de corte. A vantagem é clara, mas modesta, e em recall e F1 ele praticamente empata com a Random Forest.

**Métricas priorizadas:** 
**Por que ROC-AUC e PR-AUC, e não acurácia.**
Apenas 1,7% dos registros (616 de 36.457) são maus pagadores, cerca de 1 para cada 58 bons. Nesse cenário, um modelo que aprova todos os clientes acerta 98,3% e não encontra nenhum inadimplente, então a acurácia não distingue um bom modelo de um inútil. Por isso a comparação usa duas métricas que não dependem de um corte de probabilidade:

- **ROC-AUC:** mede se o modelo coloca os maus pagadores acima dos bons na lista de risco (0,5 equivale ao acaso).
- **PR-AUC (precisão × recall):** olha só a classe rara e responde a quantos dos clientes apontados como arriscados são de fato maus e quantos dos maus são encontrados. Seu valor de referência é a própria taxa de maus pagadores (0,017). É a métrica mais exigente para base desbalanceada e foi o critério principal de escolha.

Recall e F1 no corte padrão de 0,50 foram reportados, mas não decidiram a escolha: com tão poucos positivos, eles dependem mais de como cada modelo escala suas probabilidades (por exemplo, o uso de pesos de classe) do que da qualidade da ordenação. A regressão logística com pesos tem recall mais alto, mas o pior ROC-AUC dos três modelos treinados.

**Custo de cada tipo de erro.**

- **Falso negativo (mau pagador classificado como bom):** o banco concede o crédito e perde o valor inadimplido. É o erro mais caro por ocorrência.
- **Falso positivo (bom pagador classificado como mau):** o banco recusa ou restringe um cliente que pagaria, e perde o lucro da operação e a relação com ele. Custa menos por ocorrência, mas há 58 bons para cada mau, então o volume de falsos positivos cresce rápido quando se tenta capturar mais inadimplentes.

Como os dois erros têm custos diferentes e o equilíbrio entre eles depende de dados do banco (custo do calote e ação aplicada ao cliente sinalizado), o modelo foi escolhido pela qualidade do ranking de risco. 

---

## 6. Principais conclusões

1. **O modelo ajuda a priorizar clientes, mas não a decidir sozinho.** Ao olhar com mais atenção os 10% de clientes que o modelo considera mais arriscados, o banco encontra 23% dos maus pagadores (28 de 124), mais que o dobro dos cerca de 10% que uma escolha ao acaso encontraria. Mesmo assim, só 3,8% dessa lista são de fato maus pagadores, e para cada mau encontrado entram cerca de 25 bons clientes. Na prática, o modelo serve para direcionar o esforço de análise, não para aprovar ou negar crédito automaticamente.

2. **Recusar crédito com base no modelo não compensa financeiramente.** Sob a hipótese ilustrativa de que um calote custa 10 vezes o lucro de um bom cliente, recusar só compensaria se mais de 9,1% dos sinalizados fossem maus pagadores. Nenhuma fila testada chega a isso (a melhor, de 2% dos clientes, tem 8,2%). O resultado depende do custo real do calote, que o banco precisa informar, e a ação mais indicada para quem é sinalizado é leve: análise manual, comprovação de renda ou limite inicial menor.

3. **As variáveis que mais influenciam o modelo são cadastrais e nenhuma é decisiva sozinha.** As cinco de maior peso são estado civil, número de adultos na família, tipo de renda, tempo de emprego e ocupação, e juntas respondem por cerca de 68% da contribuição. Entre as que conseguimos observar, viúvos (cerca de 2,9% de maus pagadores) e solteiros (cerca de 2,1%) têm mais risco que casados e separados (cerca de 1,5%), e pensionistas (cerca de 2,1%) têm mais risco que servidores públicos (cerca de 1,2%). As diferenças são modestas: o grupo de maior risco tem o dobro do risco do menor, mas ainda é uma taxa baixa, então estas características ajudam a ordenar os clientes sem separar bons e maus com clareza.

4. **Os resultados são um ponto de partida, e o próximo passo é testar e enriquecer o modelo.** O teste tem poucos maus pagadores (124), os dados são só cadastrais e não há validação ao longo do tempo, então os números têm margem de erro grande. O caminho recomendado é rodar um piloto em paralelo à política atual, sem alterar decisões, e evoluir o modelo com histórico de pagamento e dados de bureau, que costumam ser os sinais que mais separam bons e maus pagadores.


### Limitações e próximos passos

**Limitações**

- **Poder de separação modesto.** O ROC-AUC é de 0,62: o modelo ordena melhor que o acaso, mas não o bastante para decidir crédito sozinho.
- **Muitos falsos alarmes.** Com apenas 1,7% de maus pagadores, cerca de 96% dos clientes da fila de maior risco (10%) são bons pagadores.
- **Margem de erro grande.** O teste tem só 124 maus pagadores, e o desempenho variou bastante entre as partes da validação cruzada.
- **Dados limitados.** A base é só cadastral, sem histórico de pagamento nem dados de bureau, e não houve validação ao longo do tempo.
- **Custo do calote hipotético.** A simulação financeira assume que um calote custa 10 vezes o lucro de um bom cliente, e com custos reais a conclusão pode mudar.
- **Variáveis pessoais.** O estado civil é a variável de maior peso, e seu uso em decisão de crédito exige validação jurídica.

**Próximos passos**

1. Incluir histórico de pagamento e dados de bureau, a melhoria com maior potencial.
2. Obter o custo real do calote para calibrar o tamanho da fila e o corte.
3. Testar o modelo sem estado civil e sem número de adultos, e validar o uso dessas variáveis com Jurídico e Compliance.
4. Validar ao longo do tempo e rodar um piloto em paralelo à política atual antes de ativar qualquer ação.
---

## 7. Estrutura do repositório

```
.
├── data/          dados brutos (raw) e tratados (processed) — não versionados
├── notebooks/     análise em ordem numerada
└── docs/          apresentação executiva
```

Detalhes e convenções em [`ESTRUTURA.md`](ESTRUTURA.md).
Antes de enviar, percorra o [`CHECKLIST.md`](CHECKLIST.md).

---

## 8. Tecnologias

- **Python 3.11** e **Jupyter Notebook**, para o desenvolvimento e a documentação das análises.
- **pandas** e **NumPy**, para leitura, tratamento e cálculos sobre os dados.
- **scikit-learn**, para o pipeline de pré-processamento (codificação das variáveis categóricas), a validação cruzada agrupada por cliente (`StratifiedGroupKFold`), os modelos de comparação (Dummy, Regressão Logística e Random Forest), as métricas (ROC-AUC, PR-AUC, precisão, recall, F1, matriz de confusão) e a importância por permutação.
- **XGBoost**, o modelo escolhido para a predição de bons e maus pagadores.
- **Matplotlib** e **seaborn**, para os gráficos de curvas, matrizes de confusão e importância das variáveis.
