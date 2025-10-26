import os
import pydicom
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, zoom, gaussian_laplace
from skimage.filters import gabor
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.morphology import remove_small_objects, closing, disk
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from typing import Optional, Tuple, List, Union
import matplotlib.pyplot as plt

# A dependência 'bm3d' é opcional. Tentamos importá-la.
try:
    from bm3d import bm3d, BM3DProfile
    BM3D_AVAILABLE = True
except ImportError:
    BM3D_AVAILABLE = False

class Preprocessing:
    """Classe para encapsular os passos de pré-processamento de imagens DICOM."""

    def __init__(self,
                 hu_clipping: bool = True,
                 homogeneous_pixel_spacing: bool = True,
                 normalization: bool = True,
                 filter: Optional[str] = "NLM",
                 target_spacing: float = 1.0,
                 nlm_patch_size: int = 5,
                 nlm_patch_distance: int = 6,
                 nlm_h_factor: float = 0.8,
                 log_sigma: float = 2.0,
                 segmentation: str = 'body'):
        """
        Inicializa a classe de pré-processamento.

        Args:
            hu_clipping (bool): Se deve aplicar clipping de unidades Hounsfield (HU).
            homogeneous_pixel_spacing (bool): Se deve reamostrar para um espaçamento de pixel isotrópico.
            filter (Optional[str]): O filtro a ser aplicado. Opções: 'BM3D', 'Gaussian', 'Median', 'NLM', 'LoG'.
            target_spacing (float): O espaçamento de pixel alvo em mm para reamostragem.
            nlm_patch_size (int): Tamanho do patch para Non-Local Means (deve ser ímpar).
            nlm_patch_distance (int): Distância máxima de busca para patches similares no NLM.
            nlm_h_factor (float): Fator multiplicativo para o parâmetro h do NLM (controla força do denoising).
            log_sigma (float): Desvio padrão do Gaussian para o Laplacian of Gaussian.
        """
        if filter is not None:
            valid_filters = ['Gaussian', 'Median', 'NLM', 'LoG', 'Gabor', 'LoG']
            if BM3D_AVAILABLE:
                valid_filters.append('BM3D')
            assert filter in valid_filters, f"Filtro inválido. Escolha entre {valid_filters} ou None."

        assert segmentation in [None, 'body', 'lung'], "Segmentação deve ser None, 'body' ou 'lung'"
        self.segmentation = segmentation        
        self.filter = filter
        self.normalization = normalization
        self.homogeneous_pixel_spacing = homogeneous_pixel_spacing
        self.hu_clipping = hu_clipping
        self.target_spacing = target_spacing
        self.segmentation = segmentation
        # Parâmetros para NLM
        self.nlm_patch_size = nlm_patch_size
        self.nlm_patch_distance = nlm_patch_distance
        self.nlm_h_factor = nlm_h_factor
        
        # Parâmetros para LoG
        self.log_sigma = log_sigma

    def clean_image(self, slice_data: pydicom.FileDataset, return_mask: bool = False) -> Union[np.ndarray,  Tuple[np.ndarray, np.ndarray]]:
        """Aplica a pipeline de pré-processamento a uma fatia DICOM."""
        # 1. Converter para HU
        np_slice_hu = self.to_hu(slice_data)
        processed_slice = np_slice_hu.copy()

        if self.segmentation is not None:
            body_mask = self.segment_body(np_slice_hu)
            processed_slice[~body_mask] = np.min(np_slice_hu)
            if self.segmentation == 'lung':
                lung_mask = self.segment_lungs(np_slice_hu, body_mask)
                processed_slice[~lung_mask] = np.min(np_slice_hu)

        # 2. Reamostrar para espaçamento de pixel homogéneo
        if self.homogeneous_pixel_spacing:
            processed_slice = self.resample_spacing(processed_slice, slice_data.PixelSpacing)

        # 3. Clipping de HU
        if self.hu_clipping:
            processed_slice = self.clip_hu(processed_slice)

        # 4. Normalização para o intervalo [0, 1]
        if self.normalization:
            processed_slice = self.normalize(processed_slice)

        # 5. Aplicação de filtros
        if self.filter is not None:
            processed_slice = self.apply_filter(processed_slice)

        if return_mask:
            mask = self.lung_mask(np_slice_hu)
            return processed_slice, mask


        #print(processed_slice[100])
        print(np.shape(processed_slice))
        return processed_slice
    

    # -------- Funções Auxiliares --------

    def to_hu(self, slice_data: pydicom.FileDataset) -> np.ndarray:
        """Converte o array de pixels para unidades Hounsfield (HU)."""
        slope = getattr(slice_data, "RescaleSlope", 1)
        intercept = getattr(slice_data, "RescaleIntercept", 0)
        return slice_data.pixel_array.astype(np.float32) * slope + intercept

    def resample_spacing(self, np_slice: np.ndarray, spacing: List[float]) -> np.ndarray:
        """Reamostra a imagem para um espaçamento isotrópico."""
        zoom_factors = (spacing[0] / self.target_spacing, spacing[1] / self.target_spacing)
        return zoom(np_slice, zoom_factors, order=1)

    def clip_hu(self, np_slice: np.ndarray, lower: int = -1200, upper: int = 600) -> np.ndarray:
        """Aplica clipping aos valores de HU para um intervalo específico."""
        return np.clip(np_slice, lower, upper)

    def normalize(self, np_slice: np.ndarray) -> np.ndarray:
        """Normaliza a imagem para o intervalo [0, 1]."""
        min_val, max_val = np.min(np_slice), np.max(np_slice)
        if max_val > min_val:
            return (np_slice - min_val) / (max_val - min_val)
        return np_slice

    def lung_mask(self, np_slice: np.ndarray, threshold: int = -350) -> np.ndarray:
        """Cria uma máscara binária do pulmão com base num limiar de HU."""
        return (np_slice <= threshold).astype(np.uint8)

    def apply_filter(self, np_slice: np.ndarray) -> np.ndarray:
        """Aplica o filtro selecionado à imagem."""
        if self.filter == 'BM3D' and BM3D_AVAILABLE:
            # O sigma_psd pode precisar de ajuste dependendo do ruído da imagem
            return bm3d(np_slice, sigma_psd=0.05)
        elif self.filter == 'Gaussian':
            return gaussian_filter(np_slice, sigma=1)
        elif self.filter == 'Median':
            return median_filter(np_slice, size=3)
        elif self.filter == 'NLM' and self.segmentation == 'body':
            # Aplicar NLM apenas dentro do corpo
            body_mask = np_slice > np.min(np_slice)
            filtered = np_slice.copy()
            filtered[body_mask] = self.apply_adaptive_nlm(np_slice)[body_mask]
            filtered[~body_mask] = np.min(np_slice)
            return filtered
        elif self.filter == 'NLM':
            return self.apply_adaptive_nlm(np_slice)
        elif self.filter == 'LoG':
            return self.apply_log_filter(np_slice)
        elif self.filter == 'Gabor':
            # Os parâmetros de frequência e theta podem ser ajustados
            real, imag = gabor(np_slice, frequency=0.6)
            return np.sqrt(real**2 + imag**2)
        return np_slice

    def apply_adaptive_nlm(self, np_slice: np.ndarray) -> np.ndarray:
        """
        Aplica Non-Local Means (NLM) adaptativo.
        O parâmetro h é estimado automaticamente com base no ruído da imagem.
        """
        # Estimar o desvio padrão do ruído na imagem
        sigma_est = estimate_sigma(np_slice, average_sigmas=True)
        
        # Calcular h adaptativamente: h = h_factor * sigma_est
        # Valores típicos de h_factor: 0.6-1.2 (quanto maior, mais suavização)
        h = self.nlm_h_factor * sigma_est
        print(f"h (adaptativo): {h}")
        
        # Aplicar Non-Local Means denoising
        # patch_size: tamanho do patch (deve ser ímpar)
        # patch_distance: distância máxima de busca para patches similares
        denoised = denoise_nl_means(
            np_slice,
            h=h,
            patch_size=self.nlm_patch_size,
            patch_distance=self.nlm_patch_distance,
            fast_mode=False,  # Usa uma versão mais precisa do algoritmo
            preserve_range=True
        )
        
        return denoised

    def apply_log_filter(self, np_slice: np.ndarray) -> np.ndarray:
        """
        Aplica Laplacian of Gaussian (LoG) como filtro.
        O LoG é útil para deteção de bordas e realce de estruturas.
        Retorna o valor absoluto para visualização.
        """
        # Aplicar Laplacian of Gaussian
        log_result = gaussian_laplace(np_slice, sigma=self.log_sigma)
        
        # Retornar valor absoluto normalizado
        # Nota: LoG pode produzir valores negativos e positivos
        # O valor absoluto realça todas as bordas independentemente da direção
        log_abs = np.abs(log_result)
        
        # Normalizar para [0, 1]
        if np.max(log_abs) > np.min(log_abs):
            log_abs = (log_abs - np.min(log_abs)) / (np.max(log_abs) - np.min(log_abs))
        
        return log_abs
    
    def segment_body(self, ct):
        body_mask = ct > -500
        body_mask = remove_small_objects(body_mask, min_size=500)
        body_mask = closing(body_mask, disk(5))
        body_mask = binary_fill_holes(body_mask)
        
        return body_mask
    
    def segment_lungs(self, hu_img, body_mask):
        body_only = hu_img * body_mask
        lung_mask = (body_only > -1000) & (body_only < -400)
        lung_mask = remove_small_objects(lung_mask, min_size=500)
        lung_mask = closing(lung_mask, disk(5))
        lung_mask = binary_fill_holes(lung_mask)
        labels = label(lung_mask)
        regions = sorted(regionprops(labels), key=lambda r: r.area, reverse=True)
        if len(regions) >= 2:
            lung_mask_final = np.isin(labels, [regions[0].label, regions[1].label])
        else:
            lung_mask_final = lung_mask
        return lung_mask_final


def plot_processed_vs_original(dicom_path):
    """
    Plota lado a lado a imagem DICOM original e a processada.
    """
    ds = pydicom.dcmread(dicom_path)
    p0 = Preprocessing(segmentation='body', hu_clipping=False, homogeneous_pixel_spacing=False, filter=None, normalization=False)
    p1 = Preprocessing(segmentation='body', hu_clipping=False, homogeneous_pixel_spacing=True, filter=None, normalization=False)
    p2 = Preprocessing(segmentation='body', hu_clipping=True, homogeneous_pixel_spacing=True, filter=None, normalization=False)
    p3 = Preprocessing(segmentation='body', hu_clipping=True, homogeneous_pixel_spacing=True, filter=None, normalization=True)
    p4 = Preprocessing(segmentation='body', hu_clipping=True, homogeneous_pixel_spacing=True, filter='NLM', normalization=True)



    fig, axes = plt.subplots(1, 6, figsize=(10, 5))
    axes[0].imshow(ds.pixel_array.astype(np.float32), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(p0.clean_image(ds), cmap='gray')
    axes[1].set_title('Body Segmentation')
    axes[1].axis('off')

    axes[2].imshow(p1.clean_image(ds), cmap='gray')
    axes[2].set_title('Homogeneous Pixel Spacing')
    axes[2].axis('off')

    axes[3].imshow(p2.clean_image(ds), cmap='gray')
    axes[3].set_title('HU Clipping')
    axes[3].axis('off')

    axes[4].imshow(p3.clean_image(ds), cmap='gray')
    axes[4].set_title('Normalization')
    axes[4].axis('off')

    axes[5].imshow(p4.clean_image(ds), cmap='gray')
    axes[5].set_title('NLM Filter')
    axes[5].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file = os.path.join(os.getcwd(), r"dataset_partition_1_5\LIDC-IDRI-0001\01-01-2000-NA-NA-30178\3000566.000000-NA-03192\1-053.dcm")

    ds = pydicom.dcmread(file)
    plot_processed_vs_original(file)