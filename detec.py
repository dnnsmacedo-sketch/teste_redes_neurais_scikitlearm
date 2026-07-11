import os
import sys

# Força o Python a usar UTF-8 para evitar erros de caracteres no Windows
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, TextVectorization
from tensorflow.keras.models import Sequential

# Configurações da página do Streamlit
st.set_page_config(page_title="Detector de Spam AI", page_icon="🔒", layout="centered")

@st.cache_data
def load_and_prepare_data():
    """Gera uma base de dados focada e equilibrada em Português."""
    dados = [
        # --- SPAMS (Label 1) ---
        {"label": 1, "message": "você ganhou 1milhão de dolares!"},
        {"label": 1, "message": "Voce ganhou 1 milhao de dolares clique no link"},
        {"label": 1, "message": "URGENTE: Seu CPF possui pendencias financeiras. Acesse para regularizar"},
        {"label": 1, "message": "Parabens! Seu numero foi sorteado para receber um Pix de 5000 reais"},
        {"label": 1, "message": "Ganhe dinheiro trabalhando de casa apenas 2 horas por dia link disponivel"},
        {"label": 1, "message": "Aviso: Sua conta sera bloqueada se nao atualizar seus dados agora"},
        {"label": 1, "message": "Voce recebeu uma transferencia pendente. Clique para aceitar"},
        {"label": 1, "message": "Desconto exclusivo de 90% apenas hoje no site confira o link"},
        {"label": 1, "message": "Seu cartao de credito foi clonado. Clique aqui para cancelar o gasto"},
        {"label": 1, "message": "Parabéns! Você ganhou um prêmio, clique aqui e resgate"},
        {"label": 1, "message": "Empréstimo liberado sem consulta ao SPC. Pegue seu dinheiro no link"},
        {"label": 1, "message": "Acesse o link para atualizar seu token de segurança do banco"},
        
        # --- MENSAGENS SEGURAS / HAM (Label 0) ---
        {"label": 0, "message": "Oi, tudo bem? Você vai para a aula hoje?"},
        {"label": 0, "message": "Oi mãe, esqueci meu casaco na sua casa, avisa quando ver"},
        {"label": 0, "message": "Por favor, me envia o relatório da reunião por e-mail até amanhã"},
        {"label": 0, "message": "Estou chegando em casa, pode colocar a janta?"},
        {"label": 0, "message": "Vamos almoçar juntos mais tarde naquele restaurante perto do trabalho?"},
        {"label": 0, "message": "Não se esqueça de comprar o pão quando voltar do serviço"},
        {"label": 0, "message": "Obrigado pela ajuda ontem, o projeto deu super certo!"},
        {"label": 0, "message": "Oi cara, você viu o jogo de futebol ontem à noite? Que jogaço"},
        {"label": 0, "message": "O aniversário da Ana vai ser no sábado, você vai?"},
        {"label": 0, "message": "Alô, estou em uma ligação de trabalho agora, te ligo depois"}
    ]
    
    df = pd.DataFrame(dados)
    return df["message"].values, df["label"].values

# --- Inicialização ---
X, y = load_and_prepare_data()

max_features = 1000
sequence_length = 50

# Vetorizador ajustado
vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)
vectorize_layer.adapt(X)

MODEL_WEIGHTS_PATH = "spam_detector_pt_weights.weights.h5"

# Construção da Arquitetura Neural
model = Sequential([
    Embedding(max_features, 16, input_shape=(sequence_length,)),
    GlobalAveragePooling1D(),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Interface Gráfica
st.title("🔒 Detector de Spam Inteligente")
st.write("Insira uma mensagem abaixo para verificar se ela é maliciosa ou segura.")

# Gerenciamento do treino/pesos
if not os.path.exists(MODEL_WEIGHTS_PATH):
    with st.spinner("Treinando e calibrando inteligência em português... Aguarde."):
        X_vectorized = vectorize_layer(X).numpy()
        # Treinamos por 50 épocas já que a base é leve e focada
        model.fit(X_vectorized, y, epochs=50, batch_size=4, verbose=0)
        model.save_weights(MODEL_WEIGHTS_PATH)
    st.success("Modelo calibrado com sucesso!")
else:
    model.load_weights(MODEL_WEIGHTS_PATH)

# Input do usuário
user_input = st.text_area("Mensagem:", placeholder="Cole o texto do SMS ou e-mail aqui...", height=150)

if st.button("Analisar Mensagem", type="primary"):
    if user_input.strip() == "":
        st.warning("Por favor, digite alguma mensagem antes de analisar.")
    else:
        # Predição
        input_vectorized = vectorize_layer(np.array([user_input])).numpy()
        prediction = model.predict(input_vectorized, verbose=0)[0][0]

        st.subheader("Resultado da Análise:")

        # Decisão baseada em probabilidade
        if prediction > 0.5:
            st.error(f"🚨 **Alerta de Spam!** (Confiança: {prediction * 100:.2f}%)")
            st.markdown("Esta mensagem possui padrões comuns de golpes, phishing ou links suspeitos.")
        else:
            st.success(f"✅ **Mensagem Segura.** (Chance de spam: {prediction * 100:.2f}%)")
            st.markdown("Não detectamos padrões alarmantes nesta mensagem.")