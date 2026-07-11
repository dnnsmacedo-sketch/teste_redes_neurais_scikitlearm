import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier

# Configuração Executiva do Dashboard
st.set_page_config(
    page_title="Hub de IA Corporativa",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 HUB DE MODELAGEM PREDITIVA E INTELIGÊNCIA ARTIFICIAL")
st.subheader("Laboratório Prático de Machine Learning para Tomada de Decisão")
st.markdown("---")

# Menu Lateral para Navegação entre os 10 Modelos
st.sidebar.title("Navegação de Modelos")
opcao = st.sidebar.radio(
    "Escolha o Desafio Analítico:",
    [
        "1. IA das Notas Escolares",
        "2. Detector de Sono Gamer",
        "3. IA do Sorvete",
        "4. Detector de Aprovação Ninja",
        "5. IA do Pet Feliz",
        "6. Detector de Filme Bom",
        "7. IA da Pizza",
        "8. Detector de Música Viral",
        "9. IA da Energia do Café",
        "10. Rede Neural dos Super-Heróis"
    ]
)

# --- FUNÇÃO AUXILIAR PARA EVITAR CRASH NO COMPILADOR DA NUVEM ---
def prever_linear(df_dados, x_col, y_col, valor_input):
    X_train = df_dados[[x_col]].values
    y_train = df_dados[y_col].values
    modelo_lr = LinearRegression()
    modelo_lr.fit(X_train, y_train)
    pred = float(modelo_lr.predict([[float(valor_input)]])[0])
    return pred

# -------------------------------------------------------------
# 1. IA DAS NOTAS ESCOLARES
# -------------------------------------------------------------
if opcao == "1. IA das Notas Escolares":
    st.header("📈 1. IA das Notas Escolares")
    st.markdown("**Objetivo:** Prever nota final acadêmica baseada nas horas dedicadas ao estudo.")
    
    df = pd.DataFrame({
        'horas': [2, 4, 5, 7, 9, 10],
        'notas': [1, 2, 4, 6, 8, 10]
    })
    
    st.dataframe(df)
    
    horas_input = st.slider("Selecione a quantidade de Horas de Estudo:", 0, 12, 6)
    predicao = prever_linear(df, 'horas', 'notas', horas_input)
    st.success(f"🎯 **Nota Prevista pela IA:** {min(max(predicao, 0.0), 10.0):.2f}")

# -------------------------------------------------------------
# 2. DETECTOR DE SONO GAMER
# -------------------------------------------------------------
elif opcao == "2. Detector de Sono Gamer":
    st.header("🎮 2. Detector de Sono Gamer")
    st.markdown("**Objetivo:** Prever o nível de cansaço acumulado baseado em horas jogando.")
    
    df = pd.DataFrame({
        'horas_jogo': [1, 2, 4, 6, 8, 10],
        'cansaco': [1, 2, 3, 5, 8, 10]
    })
    st.dataframe(df)
    
    horas_input = st.slider("Horas Jogando continuamente:", 0, 15, 5)
    predicao = prever_linear(df, 'horas_jogo', 'cansaco', horas_input)
    st.info(f"💤 **Nível de Cansaço Estimado (0 a 10):** {min(max(predicao, 0.0), 10.0):.1f}")

# -------------------------------------------------------------
# 3. IA DO SORVETE
# -------------------------------------------------------------
elif opcao == "3. IA do Sorvete":
    st.header("🍦 3. IA do Sorvete")
    st.markdown("**Objetivo:** Prever volume de vendas de sorvetes baseado na variação da temperatura.")
    
    df = pd.DataFrame({
        'temperatura': [18, 20, 24, 27, 30, 35],
        'vendas': [20, 25, 40, 55, 70, 100]
    })
    st.dataframe(df)
    
    temp_input = st.slider("Temperatura Externa (°C):", 15, 42, 28)
    predicao = prever_linear(df, 'temperatura', 'vendas', temp_input)
    st.success(f"🛒 **Previsão de Vendas:** {max(int(predicao), 0)} unidades vendidas")

# -------------------------------------------------------------
# 4. DETECTOR DE APROVAÇÃO NINJA
# -------------------------------------------------------------
elif opcao == "4. Detector de Aprovação Ninja":
    st.header("🥷 4. Detector de Aprovação Ninja")
    st.markdown("**Objetivo:** Classificar chances de aprovação acadêmica baseado nas faltas acumuladas.")
    
    df = pd.DataFrame({
        'faltas': [0, 1, 2, 5, 7, 10],
        'resultado': [1, 1, 1, 0, 0, 0]
    })
    st.dataframe(df)
    
    X = df[['faltas']].values
    y = df['resultado'].values
    modelo = LogisticRegression().fit(X, y)
    
    faltas_input = st.number_input("Insira o total de Faltas do Aluno:", min_value=0, max_value=20, value=3)
    predicao = int(modelo.predict([[float(faltas_input)]])[0])
    probabilidade = float(modelo.predict_proba([[float(faltas_input)]])[0][1]) * 100
    
    if predicao == 1:
        st.success(f"🟢 **Status Previsto:** APROVADO ({probabilidade:.1f}% de chance)")
    else:
        st.error(f"🔴 **Status Previsto:** REPROVADO ({100 - probabilidade:.1f}% de chance de reprovar)")

# -------------------------------------------------------------
# 5. IA DO PET FELIZ
# -------------------------------------------------------------
elif opcao == "5. IA do Pet Feliz":
    st.header("🐕 5. IA do Pet Feliz")
    st.markdown("**Objetivo:** Estimar o índice de felicidade do cão baseado no volume de passeios diários/semanais.")
    
    df = pd.DataFrame({
        'passeios': [1, 2, 3, 4, 5],
        'felicidade': [2, 4, 5, 8, 10]
    })
    st.dataframe(df)
    
    passeios_input = st.slider("Total de passeios efetuados:", 0, 7, 3)
    predicao = prever_linear(df, 'passeios', 'felicidade', passeios_input)
    st.info(f"🦴 **Índice de Felicidade do Pet (0 a 10):** {min(max(predicao, 0.0), 10.0):.1f}")

# -------------------------------------------------------------
# 6. DETECTOR DE FILME BOM
# -------------------------------------------------------------
elif opcao == "6. Detector de Filme Bom":
    st.header("🎬 6. Detector de Filme Bom")
    st.markdown("**Objetivo:** Estimar a nota de avaliação de obras audiovisuais com base no tempo de duração.")
    
    df = pd.DataFrame({
        'duracao': [80, 90, 100, 110, 120],
        'nota': [4, 5, 7, 8, 9]
    })
    st.dataframe(df)
    
    duracao_input = st.slider("Tempo de Duração do Filme (Minutos):", 60, 180, 105)
    predicao = prever_linear(df, 'duracao', 'nota', duracao_input)
    st.success(f"⭐ **Nota Estimada (0 a 10):** {min(max(predicao, 0.0), 10.0):.1f}")

# -------------------------------------------------------------
# 7. IA DA PIZZA
# -------------------------------------------------------------
elif opcao == "7. IA da Pizza":
    st.header("🍕 7. IA da Pizza")
    st.markdown("**Objetivo:** Prever o preço final de venda da pizza com base no diâmetro (tamanho).")
    
    df = pd.DataFrame({
        'tamanho': [20, 25, 30, 35, 40],
        'preco': [20, 30, 40, 50, 60]
    })
    st.dataframe(df)
    
    tamanho_input = st.slider("Escolha o tamanho da Pizza (Diâmetro em cm):", 15, 50, 32)
    predicao = prever_linear(df, 'tamanho', 'preco', tamanho_input)
    st.warning(f"💵 **Preço Estimado Praticado:** R$ {max(predicao, 0.0):.2f}")

# -------------------------------------------------------------
# 8. DETECTOR DE MÚSICA VIRAL
# -------------------------------------------------------------
elif opcao == "8. Detector de Música Viral":
    st.header("🎵 8. Detector de Música Viral")
    st.markdown("**Objetivo:** Projetar o potencial ou pontuação de viralização da faixa com base no ritmo (BPM).")
    
    df = pd.DataFrame({
        'bpm': [80, 90, 100, 120, 140],
        'viral': [1, 2, 4, 7, 10]
    })
    st.dataframe(df)
    
    bpm_input = st.slider("Batidas Por Minuto (BPM) da Faixa:", 60, 200, 115)
    predicao = prever_linear(df, 'bpm', 'viral', bpm_input)
    st.info(f"🔥 **Pontuação de Engajamento Viral (0 a 10):** {min(max(predicao, 0.0), 10.0):.1f}")

# -------------------------------------------------------------
# 9. IA DA ENERGIA DO CAFÉ
# -------------------------------------------------------------
elif opcao == "9. IA da Energia do Café":
    st.header("☕ 9. IA da Energia do Café")
    st.markdown("**Objetivo:** Prever o pico de produtividade/energia baseado nas doses de café consumidas.")
    
    df = pd.DataFrame({
        'xicaras': [1, 2, 3, 4, 5],
        'energia': [2, 4, 6, 8, 10]
    })
    st.dataframe(df)
    
    xicaras_input = st.slider("Total de Xícaras Ingeridas:", 0, 8, 3)
    predicao = prever_linear(df, 'xicaras', 'energia', xicaras_input)
    st.success(f"⚡ **Nível de Energia Estipulado (0 a 10):** {min(max(predicao, 0.0), 10.0):.1f}")

# -------------------------------------------------------------
# 10. REDE NEURAL DOS SUPER-HERÓIS
# -------------------------------------------------------------
elif opcao == "10. Rede Neural dos Super-Heróis":
    st.header("🦸‍♂️ 10. Rede Neural dos Super-Heróis")
    st.markdown("**Objetivo:** Classificar o status do personagem (Forte ou Fraco) empregando uma Rede Neural Artificial (MLP).")
    
    df = pd.DataFrame({
        'forca': [1, 2, 3, 7, 8, 10],
        'heroi': [0, 0, 0, 1, 1, 1]
    })
    st.dataframe(df)
    
    X = df[['forca']].values
    y = df['heroi'].values
    
    modelo = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=2000, random_state=42)
    modelo.fit(X, y)
    
    forca_input = st.slider("Defina o Nível de Força do Herói (1 a 10):", 1, 10, 5)
    predicao = int(modelo.predict([[float(forca_input)]])[0])
    
    if predicao == 1:
        st.success("💪 **Classificação da Rede Neural:** HERÓI FORTE!")
    else:
        st.error("📉 **Classificação da Rede Neural:** HERÓI FRACO!")
