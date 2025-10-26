import pandas as pd
import os


csv_path = r"C:\Users\afora\OneDrive\Ambiente de Trabalho\3ano\LabIACD\lung-cancer-classifier\Densenet_CSVs\densenet_features_copia.csv"  # substitui pelo nome do teu arquivo

# Lê o CSV
df = pd.read_csv(csv_path)

# Verifica se as colunas esperadas existem
if not {'patient_id', 'nodule_id'}.issubset(df.columns):
    raise ValueError("O CSV precisa conter as colunas 'patient_id' e 'nodule_id'.")

# Agrupa e faz a média das colunas numéricas
grouped = df.groupby(['patient_id', 'nodule_id'], as_index=False).mean(numeric_only=True)
grouped= grouped.drop(columns=['slice_idx']) 
# Gera o nome do novo arquivo
dir_path = os.path.dirname(csv_path)
base_name = os.path.basename(csv_path)
new_name = f"agrupado_{base_name}"
output_path = os.path.join(dir_path, new_name)

grouped.to_csv(output_path, index=False)

print(f"Novo arquivo salvo em: {output_path}")