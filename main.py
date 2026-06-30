from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware # Adicionado para permitir o Front-end depois!

# Carrega as variáveis do arquivo .env
load_dotenv()

# Inicializando a API
app = FastAPI(title="API - Previsão de Imóveis e Consultoria IA")

# --- NOVIDADE: Configuração de CORS (Permite que o Front-end acesse essa API) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Depois podemos restringir só para o link do front-end
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregando o modelo treinado
modelo = joblib.load('modelo_preco_imoveis.pkl')

# Configurando o Google Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
modelo_ia = genai.GenerativeModel('gemini-3.1-flash-lite')

class Imovel(BaseModel):
    OverallQual: int
    OverallCond: int
    LotArea: float
    GrLivArea: float
    FullBath: int
    KitchenAbvGr: int
    BedroomAbvGr: int
    GarageCars: int
    BldgType: int 
    YearBuilt: int

class MensagemChat(BaseModel):
    preco_estimado: float
    pergunta: str
    detalhes_imovel: str
    historico: str

@app.post("/prever")
def prever_preco(imovel: Imovel):
    # A ordem aqui precisa ser EXATAMENTE a mesma do treinamento.py
    dados_entrada = np.array([[
        imovel.OverallQual, imovel.OverallCond, imovel.LotArea, imovel.GrLivArea,
        imovel.FullBath, imovel.KitchenAbvGr, imovel.BedroomAbvGr, imovel.GarageCars,
        imovel.BldgType, imovel.YearBuilt
    ]])
    preco_previsto = modelo.predict(dados_entrada)[0]
    return {"preco_estimado": round(preco_previsto, 2)}

@app.post("/consultar-ia")
def consultar_ia(mensagem: MensagemChat):
    # Prompt atualizado para a realidade Brasileira!
    prompt = f"""
    Você é um corretor de imóveis experiente e realista no Brasil. 
    O usuário está avaliando o seguinte imóvel: {mensagem.detalhes_imovel}.
    Nosso algoritmo avaliou essa casa em R$ {mensagem.preco_estimado:,.2f}.
    
    Histórico da conversa:
    {mensagem.historico}
    
    Nova pergunta do usuário: "{mensagem.pergunta}"
    
    Responda em português do Brasil de forma direta, no máximo 100 palavras. Use Reais (R$) e metros quadrados (m²).
    Não gere o texto em formato Markdown.
    """
    
    resposta = modelo_ia.generate_content(prompt)
    return {"resposta": resposta.text}