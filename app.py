# Import de Bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

@st.cache_data
def baixar_dados(tickers, inicio):
    df_cache = pd.DataFrame()
    for TICKER in tickers:
      dados = yf.download(TICKER, inicio)
      colunas = list(dados.columns)
      for _ in range(len(colunas)):
        colunas[_] = colunas[_][0]
      dados.columns = colunas
      dados['ticker'] = TICKER
      dados = (dados.reset_index()).melt(id_vars=['Date','ticker'], var_name='var', value_name='valor')
      df_cache = pd.concat([df_cache,dados])
    return df_cache

@st.cache_data
def monte_carlo(retornos_medios, covariancias, selic, num_simulacoes=100000):
    resultados_cache = np.zeros((3, num_simulacoes))
    pesos_carteira_cache = []
    num_ativos = len(retornos_medios)

    for i in range(num_simulacoes):
      pesos = np.random.random(num_ativos)
      pesos /= np.sum(pesos)
      pesos_carteira_cache.append(pesos)

      retorno_carteira = np.sum(retornos_medios*pesos) * 252
      volatilidade_carteira = np.sqrt(np.dot(pesos.T, np.dot(covariancias * 252, pesos)))
      sharpe_carteira = (retorno_carteira - selic) / volatilidade_carteira

      resultados_cache[0,i] = retorno_carteira
      resultados_cache[1,i] = volatilidade_carteira
      resultados_cache[2,i] = sharpe_carteira

    return resultados_cache, pesos_carteira_cache

warnings.filterwarnings('ignore')
st.set_page_config(layout="wide")

##### Classificando o Perfil
st.sidebar.header("Questionário de Perfil de Risco")
respostas = {}
respostas['idade'] = st.sidebar.radio("1. Qual sua idade?", ('18-30', '31-50', '51-65', '65+'))
respostas['experiencia'] = st.sidebar.radio("2. Experiência em investimentos?", ('Iniciante', 'Intermediário', 'Avançado'))
respostas['horizonte'] = st.sidebar.radio("3. Horizonte de investimento?", ('< 1 ano', '1-5 anos', '> 5 anos'))
respostas['reacao_queda'] = st.sidebar.radio("4. Reação a uma queda de 20%?", ('Venderia tudo', 'Manteria', 'Compraria mais'))
respostas['objetivo'] = st.sidebar.radio("5. Objetivo principal?", ('Preservar capital', 'Crescimento moderado', 'Ganhos agressivos'))
respostas['perda_max'] = st.sidebar.slider("6. % máximo de perda aceitável?", 0, 50, 15)

def classificar_perfil(respostas):
    score = 0
    if respostas['idade'] in ('18-30', '31-50'): score += 15
    if respostas['experiencia'] == 'Intermediário': score += 10
    elif respostas['experiencia'] == 'Avançado': score += 20
    if respostas['horizonte'] == '1-5 anos': score += 10
    elif respostas['horizonte'] == '> 5 anos': score += 20
    if respostas['reacao_queda'] == 'Manteria': score += 10
    elif respostas['reacao_queda'] == 'Compraria mais': score += 20
    if respostas['objetivo'] == 'Crescimento moderado': score += 10
    elif respostas['objetivo'] == 'Ganhos agressivos': score += 20
    if respostas['perda_max'] > 25: score += 15

    # Conservador (<30): peso_selic=0.6, peso_arriscada=0.4
    if score < 30:
        return "Conservador", 0.4, 0.6
    # Moderado (30-70): peso_arriscada=1.0
    elif 30 <= score <= 70:
        return "Moderado", 1.0, 0.0
    # Agressivo (>70): peso_arriscada=1.2
    else:
        return "Agressivo", 1.2, -0.2

# As variáveis agora são criadas no início
perfil, peso_arriscada, peso_selic = classificar_perfil(respostas)






# Configurando Interface da Página
st.title("Otimizador de Portfólio Markowitz")
st.write("Kayo Francisco da Silva | B51234")

# Introdução
st.subheader("Sobre este Projeto")
st.markdown('''
Este trabalho tem como objetivo a construção de um portfólio de investimentos
otimizado a partir de uma análise quantitativa de ativos do mercado brasileiro.
A análise usa dados históricos para calcular métricas financeiras e, por meio da
implementção da Teoria de Otimização de Carteiras de Markowitz, identificar a
alocação de ativos que maximiza o retorno ajustado ao risco.

Foi definida também uma taxa de juros livre de risco de 12% ao ano, de modo que
os ativos arriscados só são atrativos se têm um retorno esperado superior à Selic.

O questionário ao lado ajusta a alocação final entre a carteira arriscada e um
ativo livre de risco (Renda Fixa), de acordo com o seu perfil de investidor.

Foi feita uma escolha de 33 ativos de diferentes segmentos para garantir uma
base de análise diversificada. Com o auxílio de ferramentas de IA para a
categorização dos grupos, a seleção abrange os principais setores da economia,
com dados desde janeio de 2023. O objetivo foi permitir que a análise identifique
diferentes perfis de risco e retorno, possibilitando a criação de uma carteira
de investimentos equilibrada e diversa, com redução de riscos não-sistêmicos.
''')

with st.expander("Ver lista completa de ativos analisados"):
    st.markdown("""
    - **Setor Financeiro:** ITUB4.SA, BBDC4.SA, BBAS3.SA, BBSE3.SA, B3SA3.SA, BPAC11.SA
    - **Commodities:** PETR4.SA, VALE3.SA, SUZB3.SA, GGBR4.SA, PRIO3.SA
    - **Varejo e Consumo:** LREN3.SA, MGLU3.SA, ASAI3.SA, PCAR3.SA, ABEV3.SA, MELI34.SA
    - **Saúde:** HYPE3.SA, RADL3.SA, HAPV3.SA, RDOR3.SA
    - **Industrial:** WEGE3.SA, EMBR3.SA
    - **Utilities:** ELET3.SA, CPFE3.SA, SBSP3.SA, EQTL3.SA
    - **Tecnologia:** TOTS3.SA
    - **Transporte e Serviços:** AZUL4.SA, RENT3.SA, RAIL3.SA
    - **Imobiliário:** MULT3.SA, CYRE3.SA
    """)



# Download de Dados
tickers = ['ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 'BBSE3.SA', 'B3SA3.SA',
           'BPAC11.SA', 'PETR4.SA', 'VALE3.SA', 'SUZB3.SA', 'GGBR4.SA',
           'PRIO3.SA', 'LREN3.SA', 'MGLU3.SA', 'ASAI3.SA', 'PCAR3.SA',
           'ABEV3.SA', 'MELI34.SA', 'HYPE3.SA', 'RADL3.SA', 'HAPV3.SA',
           'RDOR3.SA', 'WEGE3.SA', 'EMBR3.SA', 'ELET3.SA', 'CPFE3.SA',
           'SBSP3.SA', 'EQTL3.SA', 'TOTS3.SA', 'AZUL4.SA', 'RENT3.SA',
           'RAIL3.SA', 'MULT3.SA', 'CYRE3.SA']
selic = 0.12
[INICIO] = ['2023-01-01']
df = baixar_dados(tickers, INICIO)



# Validação e Filtro de Dados
st.markdown('''
##### Validação e Filtro de Dados

Uma vez adquiridos os dados brutos, foi feita filtragem e checagem dos dados.
Como o estudo das as métricas de avaliação do ativo e concentra nos retornos
diários, o preço de fechamento (Close) foi definido como a variável de interesse
principal.

Além disso, estamos trabalhando com preços de ações, de modo que existem algumas
restrições. Sendo assim, foram feitos alguns testes para validação dos dados.
- Checagem se todos os preços são válidos (estritamente positivos)
- Checagem se todos os ativos possuem dados de preço para o período analisado
- Checagem se os dados estão em ordem cronológica, o que é necessário para o
cálculo correto dos retornos
''')

#Filtro
df = df[df['var'] == 'Close']
df = df.dropna()

#Validação
validacoes = []
if df[df['valor'] <= 0].shape[0] == 0:
  validacoes.append(('success', '✔️ Nenhum preço inválido.'))
else:
  validacoes.append(('error', 'ERRO: Preços inválidos encontrados.'))

if df[df['valor'].isna()].shape[0] == 0:
  validacoes.append(('success', '✔️ Nenhum ativo sem preço.'))
else:
  validacoes.append(('error', 'ERRO: Ativos sem preço.'))

erros = 0
for tic in list(df['ticker'].unique()):
  erro = df[df['ticker']==tic]
  if not erro['Date'].is_monotonic_increasing:
    erros += 1
if erros == 0:
  validacoes.append(('success', '✔️ Dados em ordem cronológica.'))
else:
  validacoes.append(('error', f'ERRO: {erros} ativos fora de ordem.'))

for msg_type, msg in validacoes:
  if msg_type == 'success':
    st.success(msg)
  else:
    st.error(msg)



# Conversão de Dados para Retornos Diários
dfs = []
for tic in list(df['ticker'].unique()):
  retornos = df[df['ticker'] == tic]
  retornos['valor'] = retornos['valor'].pct_change()
  retornos = retornos.dropna()
  dfs.append(retornos)
retornos = pd.concat(dfs).drop(columns=['var'])

dic = {"ativo": ['média','volatilidade','sharpe']}
for ticker in retornos['ticker'].unique():
  metricas = []
  dff = retornos[retornos['ticker'] == ticker]['valor']

  #Média
  retorno = dff.mean() * 252
  metricas.append(retorno)
  #Volatilidade
  volatilidade = dff.std() * np.sqrt(252)
  metricas.append(volatilidade)
  #Sharpe
  sharpe = (retorno - selic) / volatilidade
  metricas.append(sharpe)

  dic[ticker] = metricas

metricas = pd.DataFrame(dic).T
metricas.columns = metricas.iloc[0]
metricas = metricas[1:]
metricas.sort_values(by='sharpe',ascending=False).head()

sharpe_positivo = metricas[metricas['sharpe'] > 0]
sharpe_positivo = sharpe_positivo.sort_values(by='sharpe',ascending=False)

top_ativos = list(sharpe_positivo.index)
top_retornos = retornos[retornos['ticker'].isin(top_ativos)]
top_retornos = top_retornos.pivot_table(index='Date',columns='ticker')


st.markdown(
"""
Feito o filtro e vlidação, dos dados, os preços de fechamento foram convertidos
em retornos diários, de mofo a normalizar os dados para permitir a comparabilidade
entre ativos de diferentes ordens de grandeza de preço. Com base nos retornos
diários, calculamos as métricas fundamentais para a avaliação de desempenho
de cada ativo:
- Retorno Médio Anualizado
- Volatilidade Anualizada
- Índice de Sharpe
Naturalmente, um ativo arriscado só é atrativo se seu retorno é superior à Selic;
ou seja, se seu Índice de Sharpe é postivo, o que indca um prêmio de risco > 0.
Sendo assim, foram selecionadosos todos os ativos com IS positivo.
""")



# Otimização de Markowitz
st.subheader("Otimização de Markowitz")
# Escolha dos Ativos
top_retornos.columns = top_retornos.columns.droplevel(0)
correlacoes = top_retornos.corr()

# Selecionar os 5 ativos
carteira = []
if len(top_ativos) >= 5:
  # Começa com o ativo de maior IS
  melhor = top_ativos[0]
  carteira.append(melhor)
  candidatos = top_ativos[1:]
  # Adiciona, um a um, os 4 ativos menos correlacionados com a carteira já formada
  while len(carteira) < 5:
    correlacoes_com_carteira = correlacoes.loc[carteira, candidatos].mean()
    proximo = correlacoes_com_carteira.idxmin()
    carteira.append(proximo)
    candidatos.remove(proximo)

# Prepara os retornos apenas desses 5 ativos para a otimização
retornos_escolhidos = top_retornos[carteira]
retornos_medios = retornos_escolhidos.mean()
covariancias = retornos_escolhidos.cov()


st.markdown("""
##### Escolha dos Ativos

Já filtrados ativos com IS > 0, para garantir a efetiva diversificação, a
seleção final não se restringe apenas a este indicador. A metodologia seleciona
o ativo de maior Sharpe e adiciona os ativos subsequentes que apresentam a menor
correlação média com a carteira já formada. O objetivo é construir um portfólio
que combine ativos de alto desempenho histórico com baixa interdependência, de
modo a reduzir riscos não sistêmicos.
""")

with st.expander("Ver os 5 ativos selecionados para a carteira"):
    lista_formatada = "* " + "\n* ".join(carteira)
    st.markdown(lista_formatada)



# Otimização
num_simulacoes = 100000
resultados, pesos_carteira = monte_carlo(retornos_medios, covariancias, selic, num_simulacoes)

# Localizar a carteira de Sharpe Máximo (melhor retorno por risco)
max_sharpe_idx = np.argmax(resultados[2])
retorno_max_sharpe = resultados[0,max_sharpe_idx]
vol_max_sharpe = resultados[1,max_sharpe_idx]
pesos_max_sharpe = pesos_carteira[max_sharpe_idx]

# Localizar a carteira de Volatilidade Mínima (menor risco)
min_vol_idx = np.argmin(resultados[1])
retorno_min_vol = resultados[0,min_vol_idx]
vol_min_vol = resultados[1,min_vol_idx]
pesos_min_vol = pesos_carteira[min_vol_idx]

# Cria os DataFrames para as duas carteiras
df_carteira_sharpe = pd.DataFrame(pesos_max_sharpe, index=carteira, columns=['Peso'])
df_carteira_min_vol = pd.DataFrame(pesos_min_vol, index=carteira, columns=['Peso'])
# Combina os pesos em uma única tabela para exibição
df_pesos_combinados = pd.DataFrame(index=carteira)
df_pesos_combinados['Peso (Max Sharpe)'] = df_carteira_sharpe['Peso']
df_pesos_combinados['Peso (Min Volat)'] = df_carteira_min_vol['Peso']

# Seleciona com base no perfil
if perfil == "Conservador":
    retorno_final = retorno_min_vol
    risco_final = vol_min_vol
    df_carteira_otima = df_carteira_min_vol
else:
    retorno_final = retorno_max_sharpe
    risco_final = vol_max_sharpe
    df_carteira_otima = df_carteira_sharpe


st.markdown("""
Definido o conjunto de 5 ativos, a etapa seguinte consiste na otimização dos
pesos de alocação de cada um. Para tal, emprega-se a Simulação de Monte Carlo,
um método estocástico que permite explorar o espaço de possíveis carteiras.
Foram geradas 100.000 combinações de portfólios com pesos aleatórios. Para cada
simulação, foram calculados o retorno esperado, a volatilidade e o Índice de
Sharpe da carteira consolidada, permitindo mapear a fronteira eficiente e
identificar as alocações ótimas.

A partir dos diversos portfólios simulados, identifica-se a carteira que
maximiza o Índice de Sharpe. Esta é considerada a carteira ótima, pois representa
a alocação de capital que oferece a melhor relação de retorno por unidade de
risco assumida. Os resultados finais detalham a composição percentual exata
desta carteira, apresentando os pesos de cada ativo, bem como o retorno anual
esperado, a volatilidade e o Índice de Sharpe consolidados, culminando em uma
solução de investimento empiricamente fundamentada.
""")





st.header("Resultados da Otimização")
st.markdown("""
A seguir, são apresentados os ativos que obtiveram um Índice de Sharpe positivo
e, ao lado, os detalhes da carteira ótima de risco detalhando seu retorno,
risco e composição.
""")

col1, col2 = st.columns(2)
with col1:
    # Tabela de Sharpe Positivo
    st.subheader("Ativos com Sharpe Positivo")
    st.dataframe(sharpe_positivo.style.format({
      'média': '{:.4f}',
      'volatilidade': '{:.4f}',
      'sharpe': '{:.4f}'
    }))
with col2:
  st.subheader("Métricas da Carteira Ótima")

  # Cria duas sub-colunas
  sub_col1, sub_col2 = st.columns(2)

  with sub_col1:
    st.markdown("##### Carteira Máximo Sharpe")
    st.metric(label="Retorno Esperado", value=f"{retorno_max_sharpe*100:.2f}%")
    st.metric(label="Volatilidade", value=f"{vol_max_sharpe*100:.2f}%")

  with sub_col2:
    st.markdown("##### Carteira Mínima Volatilidade")
    st.metric(label="Retorno Esperado", value=f"{retorno_min_vol*100:.2f}%")
    st.metric(label="Volatilidade", value=f"{vol_min_vol*100:.2f}%")

  # ADICIONE A TABELA COMBINADA AQUI:
  st.subheader("Composição das Carteiras Ótimas")
  st.dataframe((df_pesos_combinados * 100).style.format({
      'Peso (Max Sharpe)': '{:.2f}%',
      'Peso (Min Volat)': '{:.2f}%'
      }))




st.header("Aplicação Prática")
st.markdown("""
Conforme suas respostas ao questionário ao lado, seu perfil é classificado como:
""")
_, col, _ = st.columns([1, 1, 1])
with col:
    st.info(perfil)

capital_investido = st.number_input(
    "Qual o capital você deseja alocar (R$)?",
    min_value=1000.0,
    value=100000.0,
    step=500.0
)

# Exibe na barra lateral
st.sidebar.subheader(f"Perfil: {perfil}")
st.sidebar.write(f"Alocação em Risco: {peso_arriscada*100:.0f}%")
st.sidebar.write(f"Alocação em Renda Fixa: {peso_selic*100:.0f}%")

# Exibir Alocação Final com base no perfil
st.subheader("Sua Alocação Final (R$)")

col1, col2 = st.columns(2)
with col1:
    st.metric(label=f"Valor em Renda Fixa (Selic) ({peso_selic*100:.0f}%)", value=f"R$ {(capital_investido * peso_selic):,.2f}")
with col2:
    st.metric(label=f"Valor na Carteira Arriscada ({peso_arriscada*100:.0f}%)", value=f"R$ {(capital_investido * peso_arriscada):,.2f}")





# MÉTRICAS FINAIS DO PORTFÓLIO
retorno_portfolio_final = (retorno_final * peso_arriscada) + (selic * peso_selic)
risco_portfolio_final = risco_final * peso_arriscada # Risco da Selic é 0

# Alocação entre arriscados
valor_carteira_arriscada = capital_investido * peso_arriscada
df_alocacao = (df_carteira_otima * valor_carteira_arriscada)
# Alocação em Selic
df_alocacao.loc['Renda Fixa (Selic)'] = capital_investido * peso_selic
df_alocacao.columns = ['Valor (R$)']
st.dataframe(df_alocacao.style.format({'Valor (R$)': 'R$ {:,.2f}'}))

# Risco e Retorno Final
st.markdown("""
##### Métricas Finais do Portfólio
""")
col_r1, col_r2 = st.columns(2)
with col_r1:
    st.metric("Retorno Esperado Final", f"{retorno_portfolio_final * 100:.2f}%")
with col_r2:
    st.metric("Risco (Volatilidade) Final", f"{risco_portfolio_final * 100:.2f}%")
