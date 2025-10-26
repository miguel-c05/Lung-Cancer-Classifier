# Diretórios

import numpy as np
np.int = np.int64
import os
import sys
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import DataLoader
import pylidc as pl
import pydicom
import pandas as pd
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# When running this file as a script (python Features/processin_class.py) the
# package-relative imports (from ..preprocessing ...) fail because the project
# root isn't a package (and the top-level folder contains a hyphen). To make
# this file runnable both as a module and as a standalone script, add the
# project root to sys.path and use absolute imports.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.preprocessing import Preprocessing
from Features.modelo_Densenet import DICOMDataset, DenseNetFeatureExtractor



base_folder = r"C:\Users\diogo\OneDrive\Documents\Lab_IACD\LIDC-IDRI\manifest-1600709154662\LIDC-IDRI"
# Output directory for per-patient CSVs
output_dir = os.path.join(project_root, "Features", "patient_features")
os.makedirs(output_dir, exist_ok=True)



def process_patient(patient_id: str, use_cuda: bool, out_dir: str):
    preprocessor = Preprocessing(normalization=True, hu_clipping=True,
                                 homogeneous_pixel_spacing=True, filter='NLM', segmentation='body')
    
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
    if scan is None:
        return None
    
    ann_clusters = scan.cluster_annotations()
    if len(ann_clusters) == 0:
        return None
    
    dcm_dir = scan.get_path_to_dicom_files()
    dcm_files = [os.path.join(dcm_dir, f) for f in os.listdir(dcm_dir) if f.endswith('.dcm')]
    dcm_files.sort(key=lambda f: pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2])

    all_images = []
    all_patient_ids = []
    all_nodule_ids = []
    all_slice_ids = []

    
    for nodule_idx, cluster in enumerate(ann_clusters, start=1):
        ann = cluster[0]
        bbox_slices = ann.bbox()
        z_start = bbox_slices[2].start
        z_stop = bbox_slices[2].stop
        dcm_selected = dcm_files[z_start:z_stop]

        for slice_idx, f in enumerate(dcm_selected, start=z_start):
            ds = pydicom.dcmread(f)
            img = preprocessor.clean_image(ds)
            all_images.append(img)
            all_patient_ids.append(patient_id)
            all_nodule_ids.append(nodule_idx)
            all_slice_ids.append(slice_idx)
    
    dataset = DICOMDataset(all_images)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    DEVICE = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    feature_extractor = DenseNetFeatureExtractor().to(DEVICE)
    feature_extractor.eval()

    all_features = []
    with torch.no_grad():
        for i, batch_images in enumerate(dataloader):
            print(f"Extraindo batch {i+1}/{len(dataloader)} para paciente {patient_id}")
            batch_images = batch_images.to(DEVICE)
            features = feature_extractor(batch_images)
            all_features.append(features.cpu())

    if not all_features:
        print(f"Nenhuma imagem válida para o paciente {patient_id}. Pulando.")
        return None

    all_features = torch.cat(all_features, dim=0).numpy()
    expected_rows = len(all_images)
    got_rows = all_features.shape[0]
    if got_rows != expected_rows:
        print(f"Aviso: número de features ({got_rows}) != número de imagens ({expected_rows}) no paciente {patient_id}. Alinhando ao mínimo comum.")
    take = min(got_rows, expected_rows)
    df = pd.DataFrame(all_features[:take])
    df.insert(0, "slice_idx", all_slice_ids[:take])
    df.insert(0, "nodule_idx", all_nodule_ids[:take])
    df.insert(0, "patient_id", all_patient_ids[:take])

    # Write per-patient CSV to avoid write contention in parallel
    csv_path = os.path.join(out_dir, f"densenet_features_{patient_id}.csv")
    df.to_csv(csv_path, index=False)
    print(f" Paciente {patient_id} processado: {len(df)} imagens -> CSV salvo em {csv_path}.")
    return csv_path

def _safe_process_patient(args):
    """Wrapper to surface exceptions and pass multiple args through executor.map"""
    patient_id, use_cuda, out_dir = args
    try:
        path = process_patient(patient_id, use_cuda, out_dir)
        return (patient_id, path, None)
    except Exception:
        import traceback
        return (patient_id, None, traceback.format_exc())


def main():
    patients = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    patients.sort()
    print(f"Encontrados {len(patients)} pacientes em '{base_folder}'.")
    
    # Decide CUDA usage: if running multiple workers, default to CPU to avoid GPU OOM
    MAX_WORKERS = max(1, os.cpu_count() or 1)
    # Reasonable cap to avoid overloading
    MAX_WORKERS = int(MAX_WORKERS * 0.8)
    use_cuda = torch.cuda.is_available() and MAX_WORKERS == 1
    if torch.cuda.is_available() and MAX_WORKERS > 1:
        print("GPU disponível, mas será usado CPU nos workers paralelos para evitar OOM. Ajuste MAX_WORKERS se necessário.")

    print(f"Iniciando processamento paralelo dos pacientes com {MAX_WORKERS} workers...")
    tasks = [(pid, use_cuda, output_dir) for pid in patients]
    produced = []
    errors = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for patient_id, path, err in tqdm(executor.map(_safe_process_patient, tasks, chunksize=1), total=len(tasks), desc="Pacientes"):
            if err:
                errors.append((patient_id, err))
            elif path:
                produced.append((patient_id, path))

    # Merge per-patient CSVs in sorted patient order to a single file
    final_csv = os.path.join(project_root,"Densenet_CSVs", "densenet_features_copia.csv")
    if os.path.exists(final_csv):
        os.remove(final_csv)
    #apagar todos os csvs anteriores
   

    first = True
    for pid in patients:
        match = [p for p in produced if p[0] == pid]
        if not match:
            continue
        _, path = match[0]
        df = pd.read_csv(path)
        df.to_csv(final_csv, mode='a', header=first, index=False)
        first = False
    print(f"CSV final consolidado em: {final_csv}")
    if errors:
        print("Alguns pacientes falharam:")
        for pid, err in errors:
            print(f"- {pid}:\n{err}")

    for file in os.listdir(output_dir):
        if file.endswith(".csv"):
            os.remove(os.path.join(output_dir, file))

if __name__ == "__main__":
    main()

