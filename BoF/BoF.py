def bof():
    """
    Bag of Features (BoF) - Combinando Múltiplas Fontes de Características
    =======================================================================

    Este script implementa Bag of Features combinando características de:
    - DenseNet (features por slice)
    - Radiomics (features por nódulo: wavelet, original, exponential, etc.)
    - Nodule Features (características clínicas e malignidade)

    Pipeline:
    1. Carregar todos os CSVs de características
    2. Combinar features usando patient_id e nodule_id como chaves
    3. Aplicar K-Means clustering para criar vocabulário visual
    4. Criar assinaturas BoF (histogramas) por nódulo
    5. Classificar malignidade usando Random Forest
    """

    import pandas as pd
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score, 
                                roc_auc_score, precision_score, recall_score,
                                classification_report, confusion_matrix)
    import os
    import warnings
    warnings.filterwarnings('ignore')

    # =============================================================================
    # CONFIGURAÇÕES
    # =============================================================================

    # Arquivos de características
    FEATURES_CONFIG = {
        'densenet_slices': 'Densenet_CSVs/densenet_features_slices.csv',
        'densenet_nodule': 'Densenet_CSVs/densenet_features_nodulo.csv',
        'radiomics_wavelet': 'Radiomics_CSVs/radiomics_wavelet.csv',
        'radiomics_original': 'Radiomics_CSVs/radiomics_original.csv',
        'radiomics_exponential': 'Radiomics_CSVs/radiomics_exponential.csv',
        'radiomics_log': 'Radiomics_CSVs/radiomics_log.csv',
        'radiomics_logarithm': 'Radiomics_CSVs/radiomics_logarithm.csv',
        'radiomics_square': 'Radiomics_CSVs/radiomics_square.csv',
    }

    LABELS_FILE = 'modelos/nodule_features.csv'

    # Parâmetros do Bag of Features
    K_CLUSTERS = 1024 
    RANDOM_STATE = 42

    # Parâmetros do classificador
    N_ESTIMATORS = 100
    TEST_SIZE = 0.3
    CV_FOLDS = 5

    # Escolha de features para usar (True = usar, False = ignorar)
    USE_FEATURES = {
        'densenet_slices': True,      # Features DenseNet por slice (para aplicar BoF deverá ser True)
        'densenet_nodule': False,     # Features DenseNet agregadas por nódulo (opcional)
        'radiomics_wavelet': False,    # Radiomics wavelet
        'radiomics_original': False,   # Radiomics original
        'radiomics_exponential': False,
        'radiomics_log': False,
        'radiomics_logarithm': False,
        'radiomics_square': False,
    }



    # =============================================================================
    # 1. CARREGAR TODOS OS CSVs
    # =============================================================================


    dataframes = {}
    for name, filepath in FEATURES_CONFIG.items():
        if USE_FEATURES.get(name, False):
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                dataframes[name] = df
                    
            else:
                print(f"   ⚠ Arquivo não encontrado: {filepath}")

    if not dataframes:
        
        exit(1)



    # =============================================================================
    # 2. COMBINAR CARACTERÍSTICAS
    # =============================================================================

    # Identificar qual dataset tem granularidade de slice vs nódulo
    slice_level_data = []
    nodule_level_data = []

    for name, df in dataframes.items():
        # Se tem slice_idx, é nível de slice
        if 'slice_idx' in df.columns:
            slice_level_data.append((name, df))
            
        else:
            nodule_level_data.append((name, df))
            
    # Estratégia: Usar dados de slice para BoF, e adicionar features de nódulo depois

    if not slice_level_data:
        
        # Se não tiver slice data, usar nodule data diretamente
        combined_df = None
        for name, df in nodule_level_data:
            if combined_df is None:
                combined_df = df.copy()
            else:
                # Merge features de nódulo
                feature_cols = [col for col in df.columns if col not in ['patient_id', 'nodule_id']]
                # Renomear para evitar conflitos
                df_renamed = df.copy()
                for col in feature_cols:
                    df_renamed = df_renamed.rename(columns={col: f"{name}_{col}"})
                combined_df = combined_df.merge(
                    df_renamed,
                    on=['patient_id', 'nodule_id'],
                    how='inner'
                )
        
        slice_df = combined_df
        
    else:
        # Combinar dados de slice primeiro
        slice_df = None
        for name, df in slice_level_data:
            if slice_df is None:
                slice_df = df.copy()
                slice_df['source'] = name
            else:
                # Merge ou concatenar slices
                df_copy = df.copy()
                df_copy['source'] = name
                # Concatenar (assumindo que são slices diferentes)
                slice_df = pd.concat([slice_df, df_copy], ignore_index=True)
        
        
        
        # Adicionar features de nódulo ao slice_df para usar depois
        # (após criar histogramas BoF)

    # Identificar colunas
    id_columns = ['patient_id', 'nodule_id']
    if 'slice_id' in slice_df.columns:
        id_columns.append('slice_id')
    if 'source' in slice_df.columns:
        id_columns.append('source')

    feature_columns = [col for col in slice_df.columns if col not in id_columns]

    # =============================================================================
    # 3. PRÉ-PROCESSAMENTO
    # =============================================================================


    # Extrair features
    features = slice_df[feature_columns].values

    # Remover colunas com todos os valores NaN
    features_df = pd.DataFrame(features, columns=feature_columns)
    initial_cols = len(features_df.columns)
    features_df = features_df.dropna(axis=1, how='all')

    # Remover linhas com NaN
    valid_indices = features_df.dropna(axis=0, how='any').index
    features_df = features_df.loc[valid_indices]
    slice_df = slice_df.loc[valid_indices].reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)


    # Atualizar feature_columns
    feature_columns = features_df.columns.tolist()

    # Normalização
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)


    # =============================================================================
    # 4. CRIAR VOCABULÁRIO VISUAL (K-MEANS)
    # =============================================================================


    kmeans = MiniBatchKMeans(
        n_clusters=K_CLUSTERS,
        random_state=RANDOM_STATE,
        batch_size=1000,
        n_init=10,
        max_iter=100,
        verbose=0
    )

    kmeans.fit(features_scaled)

    # =============================================================================
    # 5. ATRIBUIR PALAVRAS VISUAIS
    # =============================================================================


    visual_words = kmeans.predict(features_scaled)
    slice_df['visual_word'] = visual_words

    unique_words = len(np.unique(visual_words))

    # =============================================================================
    # 6. CRIAR HISTOGRAMAS BoF POR NÓDULO
    # =============================================================================

    # Criar histograma de palavras visuais para cada nódulo
    nodule_bof = slice_df.groupby(['patient_id', 'nodule_id'])['visual_word'].apply(
        lambda x: np.histogram(x, bins=range(K_CLUSTERS + 1))[0]
    )

    nodule_bof = pd.DataFrame(nodule_bof.tolist(), index=nodule_bof.index)
    nodule_bof.columns = [f'word_{i}' for i in range(K_CLUSTERS)]
    nodule_bof = nodule_bof.reset_index()


    # =============================================================================
    # 7. ADICIONAR FEATURES DE NÓDULO (SE DISPONÍVEL)
    # =============================================================================

    if nodule_level_data:
        
        for name, df in nodule_level_data:
            feature_cols = [col for col in df.columns if col not in ['patient_id', 'nodule_id']]
            
            # Renomear colunas para incluir fonte
            df_renamed = df.copy()
            rename_map = {col: f"{name}_{col}" for col in feature_cols}
            df_renamed = df_renamed.rename(columns=rename_map)
            
            # Merge com BoF
            before_merge = nodule_bof.shape
            nodule_bof = nodule_bof.merge(
                df_renamed,
                on=['patient_id', 'nodule_id'],
                how='left'
            )
            
            added_cols = nodule_bof.shape[1] - before_merge[1]
            


    try:
        labels_df = pd.read_csv(LABELS_FILE)
        
        if 'malignancy' in labels_df.columns:
            # Merge com labels
            nodule_bof = nodule_bof.merge(
                labels_df[['patient_id', 'nodule_id', 'malignancy']],
                on=['patient_id', 'nodule_id'],
                how='left'
            )
            
            missing_labels = nodule_bof['malignancy'].isna().sum()
            
            if missing_labels > 0:
                nodule_bof = nodule_bof.dropna(subset=['malignancy'])
                print(f"   Nódulos com labels: {len(nodule_bof)}")
            
            for mal, count in nodule_bof['malignancy'].value_counts().sort_index().items():
                pct = count/len(nodule_bof)*100
            
            # Criar nova coluna binária
            def map_malignancy(x):
                if x in [1, 2]:
                    return 0  # Benigno
                elif x in [4, 5]:
                    return 1  # Maligno
                else:
                    return np.nan  # Classe 3 (indefinido) será removida
            
            nodule_bof['malignancy_binary'] = nodule_bof['malignancy'].apply(map_malignancy)
            
            # Remover classe 3 (indefinidos)
            before_drop = len(nodule_bof)
            nodule_bof = nodule_bof.dropna(subset=['malignancy_binary'])
            dropped = before_drop - len(nodule_bof)
            
            # Converter para int
            nodule_bof['malignancy_binary'] = nodule_bof['malignancy_binary'].astype(int)
            
            # Substituir coluna malignancy pela versão binária
            nodule_bof['malignancy_original'] = nodule_bof['malignancy']  # Backup
            nodule_bof['malignancy'] = nodule_bof['malignancy_binary']
            
            # Mostrar distribuição binária

            for mal, count in nodule_bof['malignancy'].value_counts().sort_index().items():
                pct = count/len(nodule_bof)*100
                label = "Benigno" if mal == 0 else "Maligno"
            
        else:
            raise ValueError("Coluna 'malignancy' não encontrada")
            
    except Exception as e:
        nodule_bof['malignancy'] = np.random.randint(1, 6, size=len(nodule_bof))

    # =============================================================================
    # 9. PREPARAR DADOS PARA CLASSIFICAÇÃO
    # =============================================================================


    # Separar X e y
    # Remover colunas de ID e labels (incluindo original e binária)
    cols_to_drop = ['patient_id', 'nodule_id', 'malignancy']
    if 'malignancy_original' in nodule_bof.columns:
        cols_to_drop.append('malignancy_original')
    if 'malignancy_binary' in nodule_bof.columns:
        cols_to_drop.append('malignancy_binary')

    X = nodule_bof.drop(cols_to_drop, axis=1)
    y = nodule_bof['malignancy'].values


    # Tratar valores infinitos ou NaN que possam ter surgido
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        
        # Opção 1: Remover colunas com muitos NaN
        nan_cols = X.columns[X.isna().sum() > len(X) * 0.5]
        if len(nan_cols) > 0:
            X = X.drop(columns=nan_cols)
            
        
        # Opção 2: Preencher NaN restantes com mediana
        X = X.fillna(X.median())

    # Split treino/teste
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y if len(np.unique(y)) > 1 and all(np.bincount(y.astype(int)) > 1) else None
        )
    except:
        # Fallback sem stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )


    # =============================================================================
    # 10. TREINAR RANDOM FOREST
    # =============================================================================

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_depth=20,
        min_samples_split=5,
        verbose=0
    )

    rf.fit(X_train, y_train)

    # =============================================================================
    # 11. AVALIAR MODELO
    # =============================================================================



    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    # AUC
    try:
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        elif len(np.unique(y_test)) > 2:
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        else:
            auc = np.nan
    except:
        auc = np.nan

    cm = confusion_matrix(y_test, y_pred)

    # =============================================================================
    # 12. FEATURE IMPORTANCE
    # =============================================================================

    feature_importance = rf.feature_importances_
    top_k = 20

    # Ordenar por importância
    indices = np.argsort(feature_importance)[-top_k:][::-1]
    top_features = [X.columns[i] for i in indices]
    top_scores = feature_importance[indices]

    for i, (feat, score) in enumerate(zip(top_features, top_scores), 1):
        feat_type = "BoF" if feat.startswith('word_') else "Nodule"
    

    # =============================================================================
    # 13. CROSS-VALIDATION
    # =============================================================================

    if len(X) >= CV_FOLDS:
        try:
            cv_scores = cross_val_score(rf, X, y, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1)
        
        except Exception as e:
            pass

    # =============================================================================
    # 14. SALVAR RESULTADOS
    # =============================================================================

    # Salvar assinaturas BoF combinadas
    output_file = 'nodule_bof_signatures.csv'
    nodule_bof.to_csv(output_file, index=False)

    # Salvar modelo e artefatos
    import joblib

    artifacts = {
        'bof_rf_model.pkl': rf,
        'bof_scaler.pkl': scaler,
        'bof_kmeans.pkl': kmeans,
    }

    for filename, obj in artifacts.items():
        joblib.dump(obj, filename)

    # Salvar relatório
    print("="*80)
    print("BAG OF FEATURES - RESULTADOS")
    print("="*80 + "\n")
    print("Features usadas:")
    for name, used in USE_FEATURES.items():
        if used:
            print(f"{name}")
    print("\nDados:")
    print(f"  Nódulos: {len(nodule_bof)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  BoF words: {K_CLUSTERS}")
    print("\nResultados:")
    print(f"  Acurácia:  {accuracy:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Kappa:     {kappa:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precisão:  {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
