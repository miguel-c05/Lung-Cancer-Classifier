import os
import re
import pandas as pd
from tqdm import tqdm
from radiomics import featureextractor

# === CONFIGURAÇÃO ===
input_folder = "NIFTI"
output_folder = "Radiomics_CSVs"
os.makedirs(output_folder, exist_ok=True)

# === CONFIGURAÇÃO DO EXTRATOR ===
params = {
    'resampledPixelSpacing': [1, 1, 1],   # resample automático 1mm
    'interpolator': 'sitkBSpline',
    'enableCExtensions': True,
    'normalize': True,
    'normalizeScale': 100
}

extractor = featureextractor.RadiomicsFeatureExtractor(**params)
extractor.enableAllFeatures()
extractor.enableImageTypes(
    Original={},
    Wavelet={},
    LoG={'sigma': [1.0, 2.0, 3.0]},
    Square={},
    SquareRoot={},
    Logarithm={},
    Exponential={}
)

# === FILTROS QUE VAMOS SEPARAR EM CSVs ===
FILTER_PREFIXES = {
    "original": "radiomics_original.csv",
    "wavelet": "radiomics_wavelet.csv",
    "log-sigma": "radiomics_log.csv",
    "square": "radiomics_square.csv",
    "squareRoot": "radiomics_squareRoot.csv",
    "logarithm": "radiomics_logarithm.csv",
    "exponential": "radiomics_exponential.csv",
}

# Dicionário para acumular resultados de cada filtro
filter_results = {f: [] for f in FILTER_PREFIXES.keys()}

# === LISTAR IMAGENS ===
files = [f for f in os.listdir(input_folder) if f.endswith("_image.nii.gz")]
files.sort()

for img_file in tqdm(files, desc="Imagens"):
    try:
        mask_file = img_file.replace("_image.nii.gz", "_mask.nii.gz")
        img_path = os.path.join(input_folder, img_file)
        mask_path = os.path.join(input_folder, mask_file)

        if not os.path.exists(mask_path):
            print(f" Máscara não encontrada para {img_file}, pulando...")
            continue

        match = re.match(r"(LIDC-IDRI-\d+)_nodule(\d+)_image\.nii\.gz", img_file)
        patient_id, nodule_id = match.group(1), match.group(2)
        print(f" Extraindo features para {patient_id} | Nódulo {nodule_id}")

       
        result = extractor.execute(img_path, mask_path)

        # Separar features por filtro
        for prefix in FILTER_PREFIXES.keys():
            subset = {
                k: v for k, v in result.items()
                if k.startswith(prefix)
            }
            if subset:
                subset["patient_id"] = patient_id
                subset["nodule_id"] = nodule_id
                filter_results[prefix].append(subset)

    except Exception as e:
        print(f" Erro ao processar {img_file}: {e}")


for prefix, rows in filter_results.items():
    if not rows:
        print(f" Nenhum dado para filtro '{prefix}'.")
        continue

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_folder, FILTER_PREFIXES[prefix])
    df.to_csv(csv_path, index=False)
    print(f" Guardado: {csv_path} ({len(df)} nódulos)")
