import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

print("1. Carregando dados...")
df = pd.read_csv('data/train.csv')

# --- NOVIDADE: CONVERSÕES ---
print("2. Convertendo medidas e moedas...")
TAXA_DOLAR = 5.20 # Cotação média 
FATOR_METROS = 0.092903 # 1 pé quadrado = 0.092903 metros quadrados

# Convertendo Dólar para Real
df['SalePrice'] = df['SalePrice'] * TAXA_DOLAR
# Convertendo Pés² para Metros²
df['LotArea'] = df['LotArea'] * FATOR_METROS
df['GrLivArea'] = df['GrLivArea'] * FATOR_METROS

# --- NOVIDADE: TRATANDO TEXTO ---
# Transformando o Tipo de Construção (BldgType) em Números
mapeamento_tipo = {
    '1Fam': 1,   # Casa Padrão
    '2fmCon': 2, # Casa dividida em 2 famílias
    'Duplex': 3, # Duplex
    'TwnhsE': 4, # Sobrado (Ponta)
    'Twnhs': 5   # Sobrado (Meio)
}
# Cria uma nova coluna numérica no dataset
df['BldgType_Num'] = df['BldgType'].map(mapeamento_tipo)

print("3. Selecionando as novas variáveis...")
features = [
    'OverallQual', 'OverallCond', 'LotArea', 'GrLivArea', 
    'FullBath', 'KitchenAbvGr', 'BedroomAbvGr', 'GarageCars', 
    'BldgType_Num', 'YearBuilt'
]
target = 'SalePrice'

X = df[features].fillna(0)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("4. Treinando modelo...")
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

precisao = r2_score(y_test, modelo.predict(X_test)) * 100
print(f"-> Precisão do Novo Modelo (R²): {precisao:.2f}%")

joblib.dump(modelo, 'modelo_preco_imoveis.pkl')
print("✅ Novo modelo 'modelo_preco_imoveis.pkl' criado com sucesso!")