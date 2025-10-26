import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

def calcular_metricas(y_true, y_pred, y_proba, is_binary=False):
    """Calcula todas as métricas pedidas.
    
    Args:
        y_true: labels verdadeiros
        y_pred: predições
        y_proba: probabilidades (ou None)
        is_binary: se True, usa average='binary' para métricas; senão usa 'macro'
    """
    avg_mode = "binary" if is_binary else "macro"
    
    metricas = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=avg_mode, zero_division=0),
        "precision": precision_score(y_true, y_pred, average=avg_mode, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=avg_mode, zero_division=0),
        #"matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
    }

    if y_proba is not None:
        # Binário: usar a coluna da classe positiva
        if isinstance(y_proba, np.ndarray) and y_proba.ndim == 2 and y_proba.shape[1] == 2:
            metricas["roc_auc_ovr"] = roc_auc_score(y_true, y_proba[:, 1])
        # Multiclasse: usar OVR com matriz de probabilidades
        elif isinstance(y_proba, np.ndarray) and y_proba.ndim == 2 and y_proba.shape[1] > 2:
            metricas["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
        else:
            metricas["roc_auc_ovr"] = roc_auc_score(y_true, y_proba)
        metricas["log_loss"] = log_loss(y_true, y_proba)
    else:
        metricas["roc_auc_ovr"] = np.nan
        metricas["log_loss"] = np.nan

    return metricas


def plotar_metricas(df_resultados, nome_df):
    """Gera gráfico comparando métricas dos modelos."""
    metricas_basicas = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "balanced_accuracy"]
    df_plot = df_resultados[[m for m in df_resultados.columns if any(x in m for x in metricas_basicas)]]

    # Extrair apenas métricas de teste
    df_plot = df_resultados[
        ["modelo"]
        + [c for c in df_resultados.columns if c.startswith("test_") and any(m in c for m in metricas_basicas)]
    ].set_index("modelo")

    df_plot.columns = [c.replace("test_", "") for c in df_plot.columns]
    df_plot.plot(kind="bar", figsize=(10, 6))
    plt.title(f"Métricas no conjunto de teste — {nome_df}")
    plt.ylabel("Valor")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plotar_roc_multiclasse(y_test, y_proba, nome_modelo, nome_df):
    """Plota curva ROC (binária ou multiclasse)."""
    if y_proba is None:
        print(f" {nome_modelo}: não suporta probabilidades — ROC não disponível.")
        return

    # Binário
    if isinstance(y_proba, np.ndarray) and y_proba.ndim == 2 and y_proba.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"ROC — {nome_modelo} ({nome_df})")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    # Multiclasse
    classes = np.unique(y_test)
    y_bin = label_binarize(y_test, classes=classes)
    n_classes = y_bin.shape[1]
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(7, 5))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f"Classe {classes[i]} (área = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"Curvas ROC — {nome_modelo} ({nome_df})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_barh_importances(feature_names, values, title):
    """Helper: barh plot for importances (values aligned with feature_names)."""
    order = np.argsort(values)
    plt.figure(figsize=(7, 4))
    plt.barh([feature_names[i] for i in order], np.asarray(values)[order])
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names, importances, nome_modelo, nome_df):
    _plot_barh_importances(feature_names, importances, f"Feature importance — {nome_modelo} ({nome_df})")


def plot_permutation_importance(feature_names, perm_mean, nome_modelo, nome_df):
    _plot_barh_importances(feature_names, perm_mean, f"Permutation importance (test) — {nome_modelo} ({nome_df})")


def plot_confusion_matrix_split(y_true, y_pred, nome_modelo, nome_df, split_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de Confusão — {nome_modelo} ({nome_df}) [{split_name}]")
    plt.tight_layout()
    plt.show()



def _feature_importances_from_model(modelo):
    """Extrai importâncias de atributos do modelo, se existir.

    - Árvores (RandomForest, XGBoost, AdaBoost): usa feature_importances_
    - Regressão logística: usa |coef_| médio entre classes
    Retorna np.ndarray ou None se não suportado.
    """
    if hasattr(modelo, "feature_importances_"):
        fi = getattr(modelo, "feature_importances_", None)
        if fi is not None:
            return np.asarray(fi)
    if isinstance(modelo, LogisticRegression) and hasattr(modelo, "coef_"):
        coef = np.asarray(modelo.coef_)
        if coef.ndim == 2:
            return np.mean(np.abs(coef), axis=0)
        elif coef.ndim == 1:
            return np.abs(coef)
    return None


# removed SHAP utilities per request


def explicar_modelos_dataframe(
    df: pd.DataFrame,
    nome_df: str,
    n_top: int = 10,
    sample_shap: int = 200,
    random_state: int = 42,
    plot: bool = False,
    # novos parâmetros de controlo
    save_metrics: bool = True,
    metrics_output_csv: str = "resultados_modelos1.csv",
    use_xai: bool = True,
    tune_hyperparams: bool = True,
    cv_folds: int = 5,
    n_iter_search: int = 20,
    bin_malignancy: bool = False,
    modelos_a_executar: list = None,
):
    """
    Treina os modelos definidos e imprime métricas de Explainable AI:
    - Feature importance do modelo (quando disponível)
    - Permutation importance (sklearn)

    Parâmetros:
    - df: DataFrame com as features e a coluna 'malignancy'
    - nome_df: nome do dataset para referência nos prints/plots
    - n_top: número de features principais a mostrar
    - sample_shap: amostras usadas para cálculo de SHAP (para acelerar)
    - random_state: semente para reprodutibilidade
    - plot: se True, mostra gráficos de barras das importâncias
    - modelos_a_executar: lista com os nomes dos modelos a executar 
                          (ex: ['random_forest', 'xgboost']). Se None, executa todos.
    Observação sobre validação: quando tune_hyperparams=True, é feita validação cruzada
    (StratifiedKFold) dentro do RandomizedSearchCV usando o conjunto de treino (X_train).
    O conjunto X_val é mantido como validação hold-out apenas para reporting de métricas.
    """
    if "malignancy" not in df.columns:
        raise ValueError("O DataFrame deve conter a coluna 'malignancy'.")

    feature_names = [c for c in df.columns if c != "malignancy" and c not in ["patient_id", "nodule_id", "identifier"]]
    
    # Binarização opcional: malignancy >= 4 é positivo
    if bin_malignancy:
        print(f"Registros antes de remover malignancy=3: {len(df)}")
        # Remover todas as linhas com malignancy = 3
        df = df[df["malignancy"] != 3].copy()
        print(f"Registros após remover malignancy=3: {len(df)}")
        
        X = df[feature_names].values
        y_raw = df["malignancy"].values
        y_num = pd.to_numeric(y_raw)
        # 0 para malignancy 1-2 (benigno), 1 para malignancy 4-5 (maligno)
        y = (y_num >= 4).astype(int)
        print(f"Distribuição binária - 0 (Benigno 1-2): {(y == 0).sum()}, 1 (Maligno 4-5): {(y == 1).sum()}")
    else:
        X = df[feature_names].values
        y_raw = df["malignancy"].values
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
    # Split 70/5/25
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=5 / 6, random_state=random_state, stratify=y_temp, shuffle=True
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    is_binary = len(np.unique(y)) == 2
    xgb_eval_metric = "logloss" if is_binary else "mlogloss"
    modelos = {
        "random_forest": RandomForestClassifier(random_state=random_state),
        "xgboost": XGBClassifier(eval_metric=xgb_eval_metric, random_state=random_state, tree_method="hist"),
        "adaboost": AdaBoostClassifier(random_state=random_state),
        "logistic_regression": LogisticRegression(max_iter=5000, random_state=random_state, solver="saga", multi_class="auto"),
        "svm": SVC(probability=True, random_state=random_state),
    }
    
    # Filtrar modelos se especificado
    if modelos_a_executar is not None:
        modelos_validos = {k: v for k, v in modelos.items() if k in modelos_a_executar}
        if not modelos_validos:
            raise ValueError(f"Nenhum modelo válido encontrado. Modelos disponíveis: {list(modelos.keys())}")
        modelos = modelos_validos
        print(f"Executando apenas os modelos: {list(modelos.keys())}")
    else:
        print(f"Executando todos os modelos: {list(modelos.keys())}")

    # espaços de procura para tuning (RandomizedSearchCV)
    param_spaces = {
        "random_forest": {
            "n_estimators": [100, 200, 300, 500, 800],
            "max_depth": [None, 5, 10, 15, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        },
        "xgboost": {
            "n_estimators": [100, 200, 400, 600],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0.0, 0.1, 0.2],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0],
        },
        "adaboost": {
            "n_estimators": [50, 100, 200, 400],
            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        },
        "logistic_regression": {
            "C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "penalty": ["l1", "l2", "elasticnet"],
            "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        },
        "svm": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["rbf", "linear", "poly", "sigmoid"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
            "degree": [2, 3, 4],
            "class_weight": [None, "balanced"],
        },
    }

    print(f"\n=== Explainable AI — {nome_df} ===")

    resultados_metricas = []

    for nome_modelo, modelo in modelos.items():
        print(f"\n-> Modelo: {nome_modelo}")

        # Tuning de hiperparâmetros (opcional)
        modelo_treinado = modelo
        if tune_hyperparams:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            search = RandomizedSearchCV(
                estimator=modelo,
                param_distributions=param_spaces[nome_modelo],
                n_iter=n_iter_search,
                scoring="balanced_accuracy",
                cv=cv,
                n_jobs=-1,
                random_state=random_state,
                verbose=0,
            )
            search.fit(X_train, y_train)
            modelo_treinado = search.best_estimator_
            print(f"   Melhor score CV (balanced_accuracy): {search.best_score_:.4f}")
            print(f"   Melhores parâmetros: {search.best_params_}")
        else:
            modelo_treinado.fit(X_train, y_train)

        # calcular métricas e (opcionalmente) guardar
        metricas = {"modelo": nome_modelo, "dataset": nome_df, "bin_malignancy": bin_malignancy}
        for split_nome, X_split, y_split in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            y_pred = modelo_treinado.predict(X_split)
            if hasattr(modelo_treinado, "predict_proba"):
                y_proba = modelo_treinado.predict_proba(X_split)
            elif hasattr(modelo_treinado, "decision_function"):
                from scipy.special import softmax
                y_proba = softmax(modelo_treinado.decision_function(X_split), axis=1)
            else:
                y_proba = None
            m = calcular_metricas(y_split, y_pred, y_proba, is_binary=bin_malignancy)
            for k, v in m.items():
                metricas[f"{split_nome}_{k}"] = v

            # Confusion matrix plots
            if plot and split_nome in ("val", "test"):
                plot_confusion_matrix_split(y_split, y_pred, nome_modelo, nome_df, split_nome)

            # ROC curves (binário ou multiclasse)
            if plot and split_nome in ("val", "test") and y_proba is not None:
                plotar_roc_multiclasse(y_split, y_proba, nome_modelo, nome_df)

        resultados_metricas.append(metricas)

        # 1) Feature importance do modelo (quando disponível)
        if not use_xai:
            continue

        fi = _feature_importances_from_model(modelo_treinado)
        if fi is not None and fi.size == len(feature_names):
            idx_top = np.argsort(fi)[::-1][:n_top]
            print("  Top features por feature_importance:")
            for rank, idx in enumerate(idx_top, 1):
                print(f"    {rank:>2}. {feature_names[idx]}: {fi[idx]:.6f}")
            if plot:
                plot_feature_importance([feature_names[i] for i in idx_top], fi[idx_top], nome_modelo, nome_df)
        else:
            print("  Feature_importance do modelo: não disponível.")

        # 2) Permutation importance (no conjunto de teste)
    
        perm = permutation_importance(modelo_treinado, X_test, y_test, n_repeats=5, random_state=random_state, scoring="balanced_accuracy")
        perm_mean = perm.importances_mean
        idx_top_perm = np.argsort(perm_mean)[::-1][:n_top]
        print("  Top features por permutation_importance (test):")
        for rank, idx in enumerate(idx_top_perm, 1):
            print(f"    {rank:>2}. {feature_names[idx]}: {perm_mean[idx]:.6f}")
        if plot:
            plot_permutation_importance([feature_names[i] for i in idx_top_perm], perm_mean[idx_top_perm], nome_modelo, nome_df)

        # SHAP analyses removed per request

    # Guardar métricas se solicitado
    if save_metrics and len(resultados_metricas) > 0:
        df_resultados = pd.DataFrame(resultados_metricas)
        try:
            antigo = pd.read_csv(metrics_output_csv)
            df_resultados = pd.concat([antigo, df_resultados], ignore_index=True)
        except FileNotFoundError:
            pass
        df_resultados.to_csv(metrics_output_csv, index=False)
        print(f"\nResultados (métricas) guardados em '{metrics_output_csv}'.")

    print("\nFim da análise de Explainable AI.\n")
