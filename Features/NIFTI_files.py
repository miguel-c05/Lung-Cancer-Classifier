# ============================================
# Script PyLIDC → NIfTI + CSV por NÓDULO (consenso ≥ 50%, resample 1x1x1mm)
# Adicionado: cálculo de volume (mm³ e mL) via uniform_cubic_resample()
# COM PROCESSAMENTO PARALELO
# ============================================

import numpy as np
np.int = np.int64
np.bool = np.bool_
np.float = np.float64
from tqdm import tqdm
import pylidc as pl
from pylidc.utils import consensus
import SimpleITK as sitk
import pandas as pd
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import traceback

# Diretórios
base_folder = "C:\\Users\\diogo\\OneDrive\\Documents\\Lab_IACD\\LIDC-IDRI\\manifest-1600709154662\\LIDC-IDRI"
output_dir = "NIFTI"
csv_path = os.path.join(output_dir, "nodule_features.csv")
os.makedirs(output_dir, exist_ok=True)
CONSENSUS_LEVEL = 0.5

# Lista de pacientes
patients = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
patients.sort()
print(f"Encontrados {len(patients)} pacientes em '{base_folder}'.")



def moda_ou_media(valores):
    if not valores:
        return None
    cnt = Counter(valores)
    max_count = max(cnt.values())
    
    candidates = [val for val, count in cnt.items() if count == max_count]
    if len(candidates) == 1:
        return candidates[0]
    else:
        
        return round(float(np.mean(candidates)))


def process_patient(patient_id):
    """
    Processa um único paciente e retorna lista de features dos nódulos.
    """
    patient_nodules = []
    
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
    

    vol = scan.to_volume()
    ann_clusters = scan.cluster_annotations()


    for i, cluster in enumerate(ann_clusters, start=1):
        print(f"  {patient_id} - Nódulo {i} com {len(cluster)} anotações")

        masks_resampled = []
        vol_ref= None
        for ann in cluster:
            vol_resampled, mask_resampled = ann.uniform_cubic_resample(
                side_length=None,
                resample_vol=True,
                resample_img=True
            )
            vol_ref = vol_resampled
            masks_resampled.append(mask_resampled.astype(np.float32))

        masks_stack = np.stack(masks_resampled, axis=0)
        consensus_fraction = np.mean(masks_stack, axis=0)
        consensus_mask = (consensus_fraction >= CONSENSUS_LEVEL).astype(np.uint8)
        volume_mm3 = float(np.sum(mask_resampled))
        volume_ml = volume_mm3 / 1000.0


        image_sitk = sitk.GetImageFromArray(vol_ref)
        mask_sitk = sitk.GetImageFromArray(consensus_mask)
        image_sitk.SetSpacing((1.0, 1.0, 1.0))
        mask_sitk.SetSpacing((1.0, 1.0, 1.0))

        # Guardar ficheiros
        img_path = os.path.join(output_dir, f"{patient_id}_nodule{i}_image.nii.gz")
        mask_path = os.path.join(output_dir, f"{patient_id}_nodule{i}_mask.nii.gz")
        sitk.WriteImage(image_sitk, img_path)
        sitk.WriteImage(mask_sitk, mask_path)

        print(f"Guardado (resample 1x1x1mm): {img_path} e {mask_path}")
        print(f"Volume estimado: {volume_mm3:.2f} mm³ ({volume_ml:.2f} mL)")
    
        # --- Extrair características de consenso ---
        fields = [
            "subtlety",
            "internalStructure",
            "calcification",
            "sphericity",
            "margin",
            "lobulation",
            "spiculation",
            "texture",
            "malignancy"
        ]
        features = {}
        for f in fields:
            valores = [getattr(a, f) for a in cluster]
            features[f] = moda_ou_media(valores)

        features["patient_id"] = patient_id
        features["nodule_id"] = i
        features["volume_mm3"] = volume_mm3

        patient_nodules.append(features)
    return patient_nodules


def main():

    MAX_WORKERS = min(os.cpu_count() or 1, 16)
    
    all_nodule_data = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(
            executor.map(process_patient, patients),
            total=len(patients),
            desc="Pacientes"
        ))
    
    for patient_nodules in results:
        all_nodule_data.extend(patient_nodules)
    
    df = pd.DataFrame(all_nodule_data)
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
