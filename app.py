import pandas as pd
import numpy as np
import sys

# Importações de bibliotecas de Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# ETAPA 1: Carregamento e Preparação dos Dados para Modelagem

# Carregando o dataset real a partir do arquivo CSV
input_filename = "dados_enriquecidos_com_alertas.csv"
try:
    # Assumindo que o separador de colunas é ponto e vírgula (;)
    df_modelo = pd.read_csv(input_filename, sep=';')
    print(f"Dataset '{input_filename}' carregado com sucesso. Shape: {df_modelo.shape}")
except FileNotFoundError:
    print(f"ERRO: Arquivo '{input_filename}' não encontrado. Verifique se o nome está correto e se ele está no mesmo diretório do script.")
    sys.exit() # Encerra o script se o arquivo não for encontrado
except Exception as e:
    print(f"Ocorreu um erro inesperado ao carregar o CSV: {e}")
    sys.exit()

# Limpeza e transformação de colunas conforme necessário
if 'capacidade' in df_modelo.columns:
    df_modelo['capacidade_num'] = df_modelo['capacidade'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    print("Coluna 'capacidade_num' criada a partir de 'capacidade'.")
else:
    print("Aviso: Coluna 'capacidade' não encontrada. A feature 'capacidade_num' não será criada.")
    # Cria a coluna com NaNs para o pipeline não quebrar, o imputer cuidará disso
    df_modelo['capacidade_num'] = np.nan


# Verificação da coluna alvo
if 'alerta_suspeita' not in df_modelo.columns:
    print("ERRO: A coluna alvo 'alerta_suspeita' não foi encontrada no CSV. Verifique o arquivo de entrada.")
    sys.exit()


print("\nContagem do alvo 'alerta_suspeita':")
print(df_modelo['alerta_suspeita'].value_counts())
print("-" * 50)

# ETAPA 2: Definição das Features (X) e do Alvo (y) e Divisão dos Dados

# Lista de todas as features esperadas pelo modelo
features_numericas_esperadas = [
    'preco', 'quantidade_vendida', 'avaliacao_nota', 'avaliacao_numero',
    'reviews_1_estrelas_pct', 'reviews_5_estrelas_pct', 'rendimento_paginas',
    'custo_por_pagina', 'capacidade_num'
]
features_categoricas_esperadas = [
    'status_vendedor', 'reputacao_cor', 'categoria_produto',
    'modelo_cartucho'
]

# Filtra para usar apenas as colunas que realmente existem no DataFrame carregado
features_numericas = [col for col in features_numericas_esperadas if col in df_modelo.columns]
features_categoricas = [col for col in features_categoricas_esperadas if col in df_modelo.columns]

print("Features numéricas que serão utilizadas:", features_numericas)
print("Features categóricas que serão utilizadas:", features_categoricas)

X = df_modelo[features_numericas + features_categoricas]
y = df_modelo['alerta_suspeita']

# Divisão em dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None)

print(f"\nDados divididos em treino ({len(X_train)} amostras) e teste ({len(X_test)} amostras).")
print("-" * 50)


# ETAPA 3: Criação do Pipeline de Pré-processamento

# Pipeline para variáveis numéricas: trata valores faltantes (NaN) com a mediana e depois aplica a padronização (scaling)
preprocessor_numerico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline para variáveis categóricas: trata valores faltantes com uma constante e depois aplica o One-Hot Encoding
preprocessor_categorico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Junta os pré-processadores
preprocessor = ColumnTransformer(
    transformers=[
        ('num', preprocessor_numerico, features_numericas),
        ('cat', preprocessor_categorico, features_categoricas)
    ],
    remainder='passthrough'
)

print("Pipeline de pré-processamento criado.")
print("-" * 50)


# ETAPA 4: Treinamento, Validação e Tuning de Modelos

# Definindo os modelos
modelos = {
    "Regressão Logística": LogisticRegression(random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

for nome, modelo in modelos.items():
    print(f"--- Treinando e Avaliando: {nome} ---")
    pipeline_modelo = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', modelo)])
    pipeline_modelo.fit(X_train, y_train)
    y_pred = pipeline_modelo.predict(X_test)
    
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Relatório de Classificação:")
    # target_names para deixar o relatório mais legível
    print(classification_report(y_test, y_pred, target_names=['Não Suspeito (0)', 'Suspeito (1)']))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 50)

# Tuning de Hiperparâmetros para o Random Forest
print("--- Iniciando Tuning de Hiperparâmetros para o Random Forest ---")

pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))])

# Grade de parâmetros para testar. Pode ser expandida, mas cuidado com o tempo de execução.
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_leaf': [1, 3]
}

# Otimizando para 'recall', que é a métrica para encontrar o máximo de suspeitos possível
grid_search = GridSearchCV(pipeline_rf, param_grid, cv=3, n_jobs=-1, verbose=1, scoring='recall')
grid_search.fit(X_train, y_train)

print(f"\nMelhores parâmetros encontrados (otimizando para Recall): {grid_search.best_params_}")
print(f"Melhor score (Recall) em validação cruzada: {grid_search.best_score_:.4f}\n")

# Avaliando o melhor modelo encontrado no conjunto de teste
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("--- Avaliação do Melhor Modelo (Random Forest Otimizado) no Conjunto de Teste ---")
print(f"Acurácia: {accuracy_score(y_test, y_pred_best):.4f}\n")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred_best, target_names=['Não Suspeito (0)', 'Suspeito (1)']))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_best))
print("-" * 50)