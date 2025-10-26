import os
import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import pandas as pd
from modelos import explicar_modelos_dataframe


def carregar_csvs_folder(folder_path):
    """Carrega todos os CSVs de um folder e retorna dicionário {nome_base: DataFrame}."""
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]
    dataframes = {}
    for f in csv_files:
        path = os.path.join(folder_path, f)
        try:
            df = pd.read_csv(path)
            nome_base = os.path.splitext(f)[0]
            dataframes[nome_base] = df
        except Exception as e:
            print(f"[ERRO] Falha a ler '{path}': {e}")
    return dataframes


def merge_por_chaves(dfs, chaves=("patient_id", "nodule_id")):
    """Faz merge interno sequencial de uma lista de DataFrames pelas chaves dadas."""
    if not dfs:
        return None
    df_merged = dfs[0]
    for nxt in dfs[1:]:
        df_merged = pd.merge(df_merged, nxt, on=list(chaves), how="inner", suffixes=("", "_dup"))
    return df_merged


def carregar_nodule_features(labels_path=None):
    """Carrega o CSV de labels (nodule_features) com colunas patient_id, nodule_id, malignancy.
    Se labels_path for None, tenta descobrir em paths comuns.
    """
    path = labels_path if labels_path else 'nodule_features.csv'
    df= pd.read_csv(path)
    return df[["patient_id", "nodule_id", "malignancy"]].copy()





def processar_combinacao(nome_combo, dfs_da_combinacao, labels_df,
                         plot=True, save_metrics=True, use_xai=False,
                         tune_hyperparams=True, bin_malignancy=True):
    """Worker: faz merge pelos ids, remove ids e corre explicar_modelos_dataframe."""
    try:

        df_merged = merge_por_chaves(dfs_da_combinacao)
        if df_merged is None or df_merged.empty:
            return {"nome": nome_combo, "status": "ignorado"}

        # Juntar labels (malignancy)
        if labels_df is None or labels_df.empty:
            return {"nome": nome_combo, "status": "erro", "erro": "Labels (nodule_features) não disponíveis"}
        df_merged = pd.merge(df_merged, labels_df, on=["patient_id", "nodule_id"], how="inner")

        # Executa modelos e explicações
        res = explicar_modelos_dataframe(
            df_merged,
            nome_df=nome_combo,
            plot=plot,
            save_metrics=save_metrics,
            use_xai=use_xai,
            tune_hyperparams=tune_hyperparams,
            bin_malignancy=bin_malignancy,
        )
        return {"nome": nome_combo, "status": "ok", "resultado": res}
    except Exception as e:
        print(f"[ERRO] {nome_combo}: {e}")
        return {"nome": nome_combo, "status": "erro", "erro": str(e)}


def executar_folder(folder_path, workers=None, labels_path=None):
    """Corre todas as combinações possíveis de CSVs num folder."""
    mapas = carregar_csvs_folder(folder_path)
    if not mapas:
        print(f"Sem CSVs em {folder_path}")
        return []

    # Carregar labels
    labels_df = carregar_nodule_features(labels_path)
    if labels_df is None:
        return []

    nomes = list(mapas.keys())
    # Gerar todas as combinações de 1..N
    combinacoes = []
    for r in range(1, len(nomes) + 1):
        for combo in itertools.combinations(nomes, r):
            combinacoes.append(combo)

    print(f"Total de combinações: {len(combinacoes)} (1..{len(nomes)})")

    # Preparar payload por combinação
    tarefas = []
    for combo in combinacoes:
        nome_combo = "||".join(combo)
        dfs_combo = [mapas[n] for n in combo]
        tarefas.append((nome_combo, dfs_combo))

    # Paralelizar em 80% dos CPUs por defeito
    if workers is None:
        cpu_count = multiprocessing.cpu_count() or 1
        workers = max(1, int(cpu_count))
    print(f"A executar em paralelo com {workers} workers...")

    resultados = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futuros = {}
        for (nome_combo, dfs_combo) in tarefas:
            # 1) bin_malignancy = True
            nome_bin = f"{nome_combo}"
            fut1 = executor.submit(
                processar_combinacao,
                nome_bin,
                dfs_combo,
                labels_df,
                False,   # plot
                True,   # save_metrics
                False,  # use_xai
                False,   # tune_hyperparams
                True,   # bin_malignancy
            )
            futuros[fut1] = nome_bin

            # 2) bin_malignancy = False
            nome_nobin = f"{nome_combo}"
            fut2 = executor.submit(
                processar_combinacao,
                nome_nobin,
                dfs_combo,
                labels_df,
                False,   # plot
                True,   # save_metrics
                False,  # use_xai
                False,   # tune_hyperparams
                False,  # bin_malignancy
            )
            futuros[fut2] = nome_nobin

        for fut in as_completed(futuros):
            nome = futuros[fut]
            try:
                resultados.append(fut.result())
            except Exception as e:
                print(f"[ERRO] Falha na combinação {nome}: {e}")
                resultados.append({"nome": nome, "status": "erro", "erro": str(e)})

    print("Execução terminada.")
    return resultados


def main():
    #parser = argparse.ArgumentParser(description="Corre modelos para todas as combinações de CSVs num folder.")
    #parser.add_argument("folder", help="Pasta com CSVs (cada CSV contém patient_id, nodule_id, features e malignancy)")
    #parser.add_argument("--workers", type=int, default=None, help="Número de workers (default: 80% dos CPUs)")
    #parser.add_argument("--labels", type=str, default=None, help="Caminho para nodule_features.csv com 'malignancy'")
    #args = parser.parse_args()
    folder = r'D:\aulas\lAB_iacd\lung-cancer-classifier\EDA1\radiomics_scaled_last'
    executar_folder(folder, workers=None, labels_path=None)


if __name__ == "__main__":
    main()