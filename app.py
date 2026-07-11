import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier

# Configuração do Dashboard
st.set_page_config(
    page_title="Hub de IA Corporativa",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 HUB DE MODELAGEM PREDITIVA E INTELIGÊNCIA ARTIFICIAL")
st.subheader("Laboratório Prático de Machine Learning para Tomada de Decisão")
st.markdown("---")

# --- FUNÇÃO AUXILIAR DE PREVISÃO ---
def prever_linear(df_dados, x_col, y_col, valor_input):
    X_train = df_dados[[x_col]].values
    y_train = df_dados[y_col].values
    modelo_lr = LinearRegression()
    modelo_lr.fit(X_train, y_train)
    pred = float(modelo_lr.predict([[float(valor_input)]])[0])
    return pred

# Criando abas limpas para navegação - Evita completamente o bug de Node do React
abas = st.tabs([
    "Notas", "Gamer", "Sorvete", "Ninja", "Pet", 
    "Filmes", "Pizza", "Música", "Café", "Heróis"
])

# -------------------------------------------------------------
# 1. IA DAS NOTAS ESCOLARES
# -------------------------------------------------------------
with abas[0]:
    st.header("📈 1. IA das Notas Escolares")
    st.markdown("**Objetivo:** Prever nota final acadêmica baseada nas horas dedicadas ao estudo.")
    df1 = pd.DataFrame({'horas': [2, 4, 5, 7, 9, 10], 'notas': [1, 2, 4, 6, 8, 10]})
    st.table(df1)
    horas_input = st.slider("Selecione a quantidade de Horas de Estudo:", 0, 12, 6, key="k1")
    pred1 = prever_linear(df1, 'horas', 'notas', horas_input)
    st.success(f"🎯 **Nota Prevista pela IA:** {min(max(pred1, 0.0), 10.0):.2f}")

# -------------------------------------------------------------
# 2. DETECTOR DE SONO GAMER
# -------------------------------------------------------------
with abas[1]:
    st.header("🎮 2. Detector de Sono Gamer")
    st.markdown("**Objetivo:** Prever o nível de cansaço acumulado baseado em horas jogando.")
    df2 = pd.DataFrame({'horas_jogo': [1, 2, 4, 6, 8, 10], 'cansaco': [1, 2, 3, 5, 8, 10]})
    st.table(df2)
    horas_jogo_input = st.slider("Horas Jogando continuamente:", 0, 15, 5, key="k2")
    pred2 = prever_linear(df2, 'horas_jogo', 'cansaco', horas_jogo_input)
    st.info(f"💤 **Nível de Cansaço Estimado (0 a 10):** {min(max(pred2, 0.0), 10.0):.1f}")

# -------------------------------------------------------------
# 3. IA DO SORVETE
# -------------------------------------------------------------
with abas[2]:
    st.header("🍦 3. IA do Sorvete")
    st.markdown("**Objetivo:** Prever volume de vendas de sorvetes baseado na variação da temperatura.")
    df3 = pd.DataFrame({'temperatura': [18, 20, 24, 27, 30, 35], 'vendas': [20, 25, 40, 55, 70, 100]})
    st.table(df3)
    temp_input = st.slider("Temperatura Externa (°C):", 15, 42, 28, key="k3")
    pred3 = prever_linear(df3, 'temperatura', 'vendas', temp_input)
    st.success(f"🛒 **Previsão de Vendas:** {max(int(pred3), 0)} unidades vendidas")

# -------------------------------------------------------------
# 4. DETECTOR DE APROVAÇÃO NINJA
# -------------------------------------------------------------
with abas[3]:
    st.header("🥷 4. Detector de Aprovação Ninja")
    st.markdown("**Objetivo:** Classificar chances de aprovação acadêmica baseado nas faltas acumuladas.")
    df4 = pd.DataFrame({'faltas': [0, 1, 2, 5, 7, 10], 'resultado': [1, 1, 1, 0, 0, 0]})
    st.table(df4)
    X4 = df4[['faltas']].values
    y4 = df4['resultado'].values
    modelo4 = LogisticRegression().fit(X4, y4)
    faltas_input = st.number_input("Insira o total de Faltas do Aluno:", min_value=0, max_value=20, value=3, key="k4")
    pred4 = int(modelo4.predict([[float(faltas_input)]])[0])
    prob4 = float(modelo4.predict_proba([[float(faltas_input)]])[0][1]) * 100
    if pred4 == 1:
        st.success(f"🟢 **Status Previsto:** APROVADO ({prob4:.1f}% de chance)")
    else:
        st.error(f"🔴 **Status Previsto:** REPROVADO ({100 - prob4:.1f}% de chance de reprovar)")

# -------------------------------------------------------------
# 5. IA DO PET FELIZ
# -------------------------------------------------------------
with abas[4]:
    st.header("🐕 5. IA do Pet Feliz")
    st.markdown("**Objetivo:** Estimar o índice de felicidade do cão baseado no volume de passeios diários/semanais.")
    df5 = pd.DataFrame({'passeios': [1, 2, 3, 4, 5], 'felicidade': [2, 4, 5, 8, 10]})
    st.table(df5)
    passeios_input = st.slider("Total de passeios efetuados:", 0, 7, 3, key="k5")
    pred5 = prever_linear(df5, 'passeios', 'felicidade', passeios_input)
    st.info(f"🦴 **Índice de Felicidade do Pet (0 a 10):** {min(max(pred5, 0.0), 10.0):.1f}")

# -------------------------------------------------------------
# 6. DETECTOR DE FILME BOM
# -------------------------------------------------------------
with abas[5]:
    st.header("🎬 6. Detector de Filme Bom")
    st.markdown("**Objetivo:** Estimar a nota de avaliação de obras audiovisuais com base no tempo de duração.")
    df6 = pd.DataFrame({'duracao': [80, 90, 100, 110, 120], 'nota': [4, 5, 7, 8, 9]})
    st.table(df6)
    duracao_input = st.slider("Tempo de Duração do Filme (Minutos):", 60, 180, 105, key="k6")
    pred6 = prever_linear(df6, 'duracao', 'nota', duracao_input)
    st.success(f"⭐ **Nota Estimada (0 a 10):** {min(max(pred6, 0.0), 10.0):.1f}")

# -------------------------------------------------------------
# 7. IA DA PIZZA
# -------------------------------------------------------------
with abas[6]:
    st.header("🍕 7. IA da Pizza")
    st.markdown("**Objetivo:** Prever o preço final de venda da pizza com base no diâmetro (tamanho).")
    df7 = pd.DataFrame({'tamanho': [20, 25, 30, 35, 40], 'preco': [20, 30, 40, 50, 60]})
    st.table(df7)
    tamanho_input = st.slider("Escolha o tamanho da Pizza (Diâmetro em cm):", 15, 50, 32, key="k7")
    pred7 = prever_linear(df7, 'tamanho', 'preco', tamanho_input)
    st.warning(f"💵 **Preço Estimado Praticado:** R$ {max(pred7, 0.0):.2f}")

# -------------------------------------------------------------
# 8. DETECTOR DE MÚSICA VIRAL
# -------------------------------------------------------------
with abas[7]:
    st.header("🎵 8. Detector de Música Viral")
    st.markdown("**Objetivo:** Projetar o potencial ou pontuação de viralização da faixa com base no ritmo (BPM).")
    df8 = pd.DataFrame({'bpm': [80, 90, 100, 120, 140], 'viral': [1, 2, 4, 7, 10]})
    st.table(df8)
    bpm_input = st.slider("Batidas Por Minuto (BPM) da Faixa:", 60, 200, 115, key="k8")
    pred8 = prever_linear(df8, 'bpm', 'viral', bpm_input)
    st.info(f"🔥 **Pontuação de Engajamento Viral (0 a 10):** {min(max(pred8, 0.0), 10.0):.1f}")

# -------------------------------------------------------------
# 9. IA DA ENERGIA DO CAFÉ
# -------------------------------------------------------------
with abas[8]:
    st.header("☕ 9. IA da Energia do Café")
    st.markdown("**Objetivo:** Prever o pico de produtividade/energia baseado nas doses de café consumidas.")
    df9 = pd.DataFrame({'xicaras': [1, 2, 3, 4, 5], 'energia': [2, 4, 6, 8, 10]})
    st.table(df9)
    xicaras_input = st.slider("Total de Xícaras Ingeridas:", 0, 8, 3, key="k9")
    pred9 = prever_linear(df9, 'xicaras', 'energia', xicaras_input)
    st.success(f"⚡ **Nível de Energia Estipulado (0 a 10):** {min(max(pred9, 0.0), 10.0):.1f}")

# -------------------------------------------------------------
# 10. REDE NEURAL DOS SUPER-HERÓIS
# -------------------------------------------------------------
with abas[9]:
    st.header("🦸‍♂️ 10. Rede Neural dos Super-Heróis")
    st.markdown("**Objetivo:** Classificar o status do personagem (Forte ou Fraco) empregando uma Rede Neural Artificial (MLP).")
    df10 = pd.DataFrame({'forca': [1, 2, 3, 7, 8, 10], 'heroi': [0, 0, 0, 1, 1, 1]})
    st.table(df10)
    X10 = df10[['forca']].values
    y10 = df10['heroi'].values
    modelo10 = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=2000, random_state=42)
    modelo10.fit(X10, y10)
    forca_input = st.slider("Defina o Nível de Força do Herói (1 a 10):", 1, 10, 5, key="k10")
    pred10 = int(modelo10.predict([[float(forca_input)]])[0])
    if pred10 == 1:
        st.success("💪 **Classificação da Rede Neural:** HERÓI FORTE!")
    else:
        st.error("📉 **Classificação da Rede Neural:** HERÓI FRACO!")
