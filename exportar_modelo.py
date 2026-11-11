import os
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Módulos de Scikit-Learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_curve,
    roc_auc_score
)

# Conector personalizado a MySQL
from src.database.connection2 import MySQLConnector

RUTA_MODELO = 'models/modelo_scouting_rf.joblib'
CARPETA_FIGURAS = 'reports/figures'


def main():
    print("=================================================================")
    print("  ACTIVIDAD FINAL: ENTRENAMIENTO, OPTIMIZACIÓN Y SERIALIZACIÓN")
    print("=================================================================\n")

    # -------------------------------------------------------------
    # FASE 1: INGESTA DESDE MYSQL
    # -------------------------------------------------------------
    db = MySQLConnector()

    query = """
        SELECT
            p.player_id,
            p.name AS jugador,
            p.position AS posicion,
            p.height_in_cm AS estatura_cm,
            TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) AS edad,
            p.market_value_in_eur AS valor_mercado
        FROM players p
        WHERE p.market_value_in_eur IS NOT NULL
          AND p.date_of_birth IS NOT NULL;
    """

    print("[1] Extrayendo datos desde MySQL...")
    df = db.extract_data(query)

    if df.empty:
        print("[ERROR] No se obtuvieron datos de la base de datos.")
        return

    # -------------------------------------------------------------
    # FASE 2: PREPROCESAMIENTO Y FEATURE ENGINEERING
    # -------------------------------------------------------------
    df['estatura_cm'] = pd.to_numeric(df['estatura_cm'], errors='coerce')
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['valor_mercado'] = pd.to_numeric(df['valor_mercado'], errors='coerce')
    df = df.dropna(subset=['valor_mercado'])

    df['posicion'] = df['posicion'].fillna('Sin Posicion')
    nulos_estatura = int(df['estatura_cm'].isna().sum())

    # Imputación por mediana agrupada (se guardan las medianas para producción)
    medianas_estatura = df.groupby('posicion')['estatura_cm'].median().to_dict()
    mediana_global_estatura = float(df['estatura_cm'].median())
    mediana_global_edad = float(df['edad'].median())

    df['estatura_cm'] = df.groupby('posicion')['estatura_cm'].transform(lambda x: x.fillna(x.median()))
    df['estatura_cm'] = df['estatura_cm'].fillna(mediana_global_estatura)
    df['edad'] = df['edad'].fillna(mediana_global_edad)

    # Target binario por percentil: Élite = Top 25% de valor comercial
    umbral_elite = float(df['valor_mercado'].quantile(0.75))
    df['es_elite'] = (df['valor_mercado'] >= umbral_elite).astype(int)

    # One-Hot Encoding de la posición
    df_dummies = pd.get_dummies(df[['posicion']], drop_first=True, dtype=int)
    X = pd.concat([df[['edad', 'estatura_cm']], df_dummies], axis=1)
    y = df['es_elite']

    # División 80/20 estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[OK] Registros procesados: {X.shape[0]} | Nulos de estatura imputados: {nulos_estatura}")
    print(f"     Umbral Élite (P75): {umbral_elite:,.0f} EUR")
    print(f"     Clase 0 (Estándar): {int((y == 0).sum())} | Clase 1 (Élite): {int((y == 1).sum())}")
    print(f"     Proporción Élite -> Train: {y_train.mean():.4f} | Test: {y_test.mean():.4f}\n")

    # -------------------------------------------------------------
    # FASE 3: DIAGNÓSTICO DE OVERFITTING Y PODA PREVIA
    # -------------------------------------------------------------
    print("[2] Diagnóstico de Overfitting (Árbol libre vs Árbol podado)...")
    tree_libre = DecisionTreeClassifier(max_depth=None, random_state=42)
    tree_libre.fit(X_train, y_train)

    tree_podado = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    tree_podado.fit(X_train, y_train)

    def resumen(modelo):
        y_pred = modelo.predict(X_test)
        return {
            'acc_train': accuracy_score(y_train, modelo.predict(X_train)),
            'acc_test': accuracy_score(y_test, y_pred),
            'precision_elite': precision_score(y_test, y_pred, zero_division=0),
            'recall_elite': recall_score(y_test, y_pred, zero_division=0),
            'f1_elite': f1_score(y_test, y_pred, zero_division=0),
        }

    m_libre = resumen(tree_libre)
    m_podado = resumen(tree_podado)
    for nombre, m in [('Árbol Sin Límite', m_libre), ('Árbol Podado', m_podado)]:
        print(f"  • {nombre}: Train {m['acc_train']:.4f} | Test {m['acc_test']:.4f} | "
              f"Brecha {m['acc_train'] - m['acc_test']:.4f} | F1 Élite {m['f1_elite']:.4f}")
    print()

    # -------------------------------------------------------------
    # FASE 4: PARADOJA DE LA EXACTITUD Y PENALIZACIÓN DE COSTO
    # -------------------------------------------------------------
    print("[3] Paradoja de la Exactitud y class_weight='balanced'...")
    dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    m_dummy = resumen(dummy)

    tree_balanceado = DecisionTreeClassifier(
        max_depth=4, class_weight='balanced', random_state=42
    ).fit(X_train, y_train)
    m_balanceado = resumen(tree_balanceado)

    print(f"  • Clasificador 'siempre Estándar': Accuracy {m_dummy['acc_test']:.4f} | Recall Élite {m_dummy['recall_elite']:.4f}")
    print(f"  • Árbol balanceado (max_depth=4): Accuracy {m_balanceado['acc_test']:.4f} | Recall Élite {m_balanceado['recall_elite']:.4f}\n")

    # -------------------------------------------------------------
    # FASE 5: RANDOM FOREST + GRIDSEARCHCV
    # -------------------------------------------------------------
    print("[4] Optimizando RandomForestClassifier con GridSearchCV (5-Fold CV)...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'min_samples_leaf': [5, 20],
        'class_weight': ['balanced'],
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    rf_optimo = grid_search.best_estimator_
    m_rf = resumen(rf_optimo)

    print(f"  • Mejores Hiperparámetros: {grid_search.best_params_}")
    print(f"  • Mejor ROC-AUC promedio en CV: {grid_search.best_score_:.4f}")
    print(f"  • Random Forest en Test: Accuracy {m_rf['acc_test']:.4f} | Recall Élite {m_rf['recall_elite']:.4f} | F1 Élite {m_rf['f1_elite']:.4f}\n")

    # -------------------------------------------------------------
    # FASE 6: CURVA ROC-AUC E IMPORTANCIA DE VARIABLES
    # -------------------------------------------------------------
    y_prob_tree = tree_balanceado.predict_proba(X_test)[:, 1]
    y_prob_rf = rf_optimo.predict_proba(X_test)[:, 1]
    auc_tree = roc_auc_score(y_test, y_prob_tree)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    auc_podado = roc_auc_score(y_test, tree_podado.predict_proba(X_test)[:, 1])

    print("[5] Comparativa ROC-AUC")
    print(f"  • Árbol de Decisión Único ROC-AUC: {auc_tree:.4f}")
    print(f"  • Random Forest Optimizado ROC-AUC: {auc_rf:.4f}\n")

    importancias = pd.Series(rf_optimo.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Importancia de Variables (Gini):")
    print(importancias.round(4).to_string())
    print()

    # -------------------------------------------------------------
    # FASE 7: GRÁFICAS PARA EL INFORME
    # -------------------------------------------------------------
    os.makedirs(CARPETA_FIGURAS, exist_ok=True)

    # Gráfico 1: Diagnóstico de Overfitting
    fig, ax = plt.subplots(figsize=(8, 5))
    etapas = ['Árbol Sin Límite', 'Árbol Podado\n(max_depth=3, min_leaf=10)']
    x = np.arange(len(etapas))
    width = 0.35
    train_accs = [m_libre['acc_train'], m_podado['acc_train']]
    test_accs = [m_libre['acc_test'], m_podado['acc_test']]
    b1 = ax.bar(x - width/2, train_accs, width, label='Train Accuracy', color='#e74c3c', edgecolor='black')
    b2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy', color='#3498db', edgecolor='black')
    ax.bar_label(b1, fmt='%.3f')
    ax.bar_label(b2, fmt='%.3f')
    ax.set_title('Diagnóstico de Overfitting y Efecto de la Poda', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(etapas)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(f'{CARPETA_FIGURAS}/01_overfitting.png', dpi=200)
    plt.close()

    # Gráfico 2: Desbalance de clases
    fig, ax = plt.subplots(figsize=(6, 4.5))
    conteo = y.value_counts().reindex([0, 1], fill_value=0)
    ax.bar(['Estándar (0)', 'Élite (1)'], conteo.values, color=['#2980b9', '#e67e22'], edgecolor='black')
    for i, v in enumerate(conteo.values):
        ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')
    ax.set_title('Distribución del Target (Top 25%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cantidad de Jugadores')
    plt.tight_layout()
    plt.savefig(f'{CARPETA_FIGURAS}/02_desbalance.png', dpi=200)
    plt.close()

    # Gráfico 3: Curvas ROC comparativas
    fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_tree, tpr_tree, color='#e67e22', linestyle='--', linewidth=2,
            label=f'Árbol Único (AUC = {auc_tree:.3f})')
    ax.plot(fpr_rf, tpr_rf, color='#27ae60', linewidth=2.5,
            label=f'Random Forest (AUC = {auc_rf:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Azar (AUC = 0.500)')
    ax.set_title('Comparativa de Curvas ROC', fontsize=13, fontweight='bold')
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR - Recall)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{CARPETA_FIGURAS}/03_roc_auc.png', dpi=200)
    plt.close()

    # Gráfico 4: Importancia de variables
    fig, ax = plt.subplots(figsize=(8, 5))
    imp_ordenada = importancias.sort_values(ascending=True)
    ax.barh(imp_ordenada.index, imp_ordenada.values, color='#2980b9', edgecolor='black', alpha=0.85)
    for i, v in enumerate(imp_ordenada.values):
        ax.text(v, i, f' {v:.3f}', va='center')
    ax.set_title('Importancia de Variables (Random Forest - Gini)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importancia Relativa (reducción de impureza Gini)')
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    plt.tight_layout()
    plt.savefig(f'{CARPETA_FIGURAS}/04_feature_importance.png', dpi=200)
    plt.close()

    # Tasa de élite por posición y rango de edad (apoyo a la interpretación)
    tasa_posicion = df.groupby('posicion')['es_elite'].mean().sort_values(ascending=False)
    rangos_edad = pd.cut(df['edad'], bins=[0, 20, 23, 26, 29, 32, 60],
                         labels=['<=20', '21-23', '24-26', '27-29', '30-32', '33+'])
    tasa_edad = df.groupby(rangos_edad, observed=True)['es_elite'].mean()

    # -------------------------------------------------------------
    # FASE 8: SERIALIZACIÓN DEL PAQUETE DE DESPLIEGUE
    # -------------------------------------------------------------
    # Se reentrena con los mejores hiperparámetros sobre el 100% de los datos
    print("[6] Reentrenando el modelo final con el 100% de los datos...")
    modelo_final = RandomForestClassifier(random_state=42, n_jobs=-1, **grid_search.best_params_)
    modelo_final.fit(X, y)

    paquete_despliegue = {
        'modelo': modelo_final,
        'columnas_X': X.columns.tolist(),
        'posiciones_validas': sorted(df['posicion'].unique().tolist()),
        'medianas_estatura': medianas_estatura,
        'mediana_global_estatura': mediana_global_estatura,
        'mediana_global_edad': mediana_global_edad,
        'umbral_elite_eur': umbral_elite,
        'mejores_parametros': grid_search.best_params_,
        'auc_test': auc_rf,
    }

    os.makedirs('models', exist_ok=True)
    joblib.dump(paquete_despliegue, RUTA_MODELO)
    print(f"[OK] Paquete de modelo exportado exitosamente en: '{RUTA_MODELO}'")

    # Métricas en JSON para el informe
    metricas = {
        'registros': int(X.shape[0]),
        'nulos_estatura_imputados': nulos_estatura,
        'umbral_elite_eur': umbral_elite,
        'clase_0': int((y == 0).sum()),
        'clase_1': int((y == 1).sum()),
        'prop_elite_train': float(y_train.mean()),
        'prop_elite_test': float(y_test.mean()),
        'medianas_estatura': medianas_estatura,
        'mediana_global_edad': mediana_global_edad,
        'arbol_libre': m_libre,
        'arbol_libre_profundidad': int(tree_libre.get_depth()),
        'arbol_libre_hojas': int(tree_libre.get_n_leaves()),
        'arbol_podado': m_podado,
        'auc_arbol_podado': auc_podado,
        'dummy_mayoritaria': m_dummy,
        'arbol_balanceado': m_balanceado,
        'random_forest': m_rf,
        'mejores_parametros': grid_search.best_params_,
        'mejor_auc_cv': grid_search.best_score_,
        'auc_arbol': auc_tree,
        'auc_rf': auc_rf,
        'importancias': importancias.to_dict(),
        'tasa_elite_posicion': tasa_posicion.to_dict(),
        'tasa_elite_edad': {str(k): float(v) for k, v in tasa_edad.items()},
    }
    with open('reports/metricas.json', 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False, default=float)
    print("[OK] Gráficas en 'reports/figures/' y métricas en 'reports/metricas.json'.")


if __name__ == "__main__":
    main()
