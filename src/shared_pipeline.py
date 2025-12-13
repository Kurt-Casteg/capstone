"""
Shared Data Pipeline for Histopathology Classification
=======================================================

Pipeline de datos estandarizado y GPU-optimizado para todos los modelos:
- CNN Baseline (150×150, color augmentation)
- VGG19 (224×224, VGG preprocessing)
- ResNet50 (224×224, ResNet preprocessing)
- EfficientNet (224×224, EfficientNet preprocessing)
- ViT (Vision Transformer) (224×224, ImageNet normalization)

DISEÑO:
- Secciones compartidas: carga, geometric augmentation
- Secciones parametrizadas: resolución, color aug, model preprocessing

USO:
    from shared_pipeline import create_datasets, CLASS_NAMES
    
    # Para CNN Baseline
    train_ds, val_ds, test_ds = create_datasets(
        data_dir='../data',
        model_type='baseline',
        img_size=150
    )
    
    # Para Transfer Learning
    train_ds, val_ds, test_ds = create_datasets(
        data_dir='../data',
        model_type='resnet',  # o 'vgg', 'efficientnet', 'vit'
        img_size=224
    )

Autor: Capstone Project
Fecha: Diciembre 2025
Compatibilidad: TensorFlow 2.10.1+
"""

import tensorflow as tf
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# =============================================================================
# CONSTANTES GLOBALES
# =============================================================================

AUTOTUNE = tf.data.AUTOTUNE

CLASS_NAMES = [
    '01_TUMOR',
    '02_STROMA', 
    '03_COMPLEX',
    '04_LYMPHO',
    '05_DEBRIS',
    '06_MUCOSA',
    '07_ADIPOSE',
    '08_EMPTY'
]


# =============================================================================
# SECCIÓN 2.1: CONFIGURACIÓN
# =============================================================================

def get_config(
    model_type: str = 'baseline',
    img_size: int = 150,
    batch_size: int = 32,
    seed: int = 42,
    data_dir: str = '../data',
    use_color_augmentation: bool = True
) -> Dict[str, Any]:
    """
    Genera configuración para el pipeline según el tipo de modelo.
    
    Args:
        model_type: 'baseline', 'vgg', 'resnet', 'efficientnet', 'vit'
        img_size: Tamaño de imagen (150 para baseline, 224 para TL)
        batch_size: Tamaño de batch
        seed: Semilla para reproducibilidad
        data_dir: Directorio de datos
        use_color_augmentation: Habilitar augmentation de color (solo baseline)
    
    Returns:
        Diccionario de configuración
    """
    # Configuración base compartida
    config = {
        'model_type': model_type.lower(),
        'img_height': img_size,
        'img_width': img_size,
        'channels': 3,
        'num_classes': 8,
        'batch_size': batch_size,
        'seed': seed,
        'data_dir': Path(data_dir),
        
        # Augmentation geométrica (ESTANDARIZADA para todos)
        'horizontal_flip': True,
        'vertical_flip': True,
        'zoom_range': 0.1,  # Zoom IN only: [1.0, 1.1]
        
        # Augmentation de color (solo para baseline CNN)
        'use_color_augmentation': use_color_augmentation and model_type.lower() == 'baseline',
        'hue_delta': 0.15,
        'saturation_lower': 0.6,
        'saturation_upper': 1.4,
        'brightness_delta': 0.25,
    }
    
    return config


# =============================================================================
# SECCIÓN 2.2: COLOR AUGMENTATION (Solo CNN Baseline)
# =============================================================================

def color_augmentation(image: tf.Tensor, config: Dict[str, Any]) -> tf.Tensor:
    """
    Aplica augmentation de color en espacio HSV.
    
    SOLO para CNN Baseline - simula variabilidad en tinción H&E.
    NO usar con modelos de Transfer Learning (interfiere con preprocessing ImageNet).
    
    Args:
        image: Tensor [H, W, 3] en rango [0, 255]
        config: Diccionario de configuración
    
    Returns:
        Imagen augmentada en rango [0, 255]
    """
    # Normalizar a [0, 1] para operaciones HSV
    image = image / 255.0
    
    # Augmentaciones HSV
    image = tf.image.random_hue(image, max_delta=config['hue_delta'])
    image = tf.image.random_saturation(
        image, 
        lower=config['saturation_lower'], 
        upper=config['saturation_upper']
    )
    image = tf.image.random_brightness(image, max_delta=config['brightness_delta'])
    
    # Clip y escalar de vuelta a [0, 255]
    image = tf.clip_by_value(image, 0.0, 1.0) * 255.0
    
    return image


# =============================================================================
# SECCIÓN 2.3: GEOMETRIC AUGMENTATION (Todos los modelos)
# =============================================================================

def geometric_augmentation(image: tf.Tensor, config: Dict[str, Any]) -> tf.Tensor:
    """
    Aplica transformaciones geométricas invariantes a rotación.
    
    ESTANDARIZADO para todos los modelos - garantiza comparación justa.
    
    Transformaciones:
    - Rotaciones de 90° (k=0,1,2,3)
    - Flip horizontal
    - Flip vertical  
    - Zoom IN only [1.0, 1.1] - evita padding negro
    
    Args:
        image: Tensor [H, W, 3]
        config: Diccionario de configuración
    
    Returns:
        Imagen transformada
    """
    # Rotaciones de 90 grados (k=0,1,2,3 para 0°,90°,180°,270°)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    
    # Flips aleatorios
    if config['horizontal_flip']:
        image = tf.image.random_flip_left_right(image)
    if config['vertical_flip']:
        image = tf.image.random_flip_up_down(image)
    
    # Zoom IN only (evita padding negro)
    if config['zoom_range'] > 0:
        zoom_factor = tf.random.uniform([], 1.0, 1.0 + config['zoom_range'])
        new_h = tf.cast(tf.cast(config['img_height'], tf.float32) * zoom_factor, tf.int32)
        new_w = tf.cast(tf.cast(config['img_width'], tf.float32) * zoom_factor, tf.int32)
        
        image = tf.image.resize(image, [new_h, new_w])
        image = tf.image.resize_with_crop_or_pad(
            image, 
            config['img_height'], 
            config['img_width']
        )
    
    return image


# =============================================================================
# SECCIÓN 2.4: CARGA DE IMÁGENES (GPU-Optimizado)
# =============================================================================

def load_image(path: tf.Tensor, config: Dict[str, Any]) -> tf.Tensor:
    """
    Carga imagen TIFF usando tensorflow-io (GPU-optimizado).
    
    Pipeline:
    1. Leer archivo TIFF
    2. Decodificar con tfio
    3. Resize a resolución objetivo
    4. Convertir a float32 [0, 255]
    
    Args:
        path: Ruta al archivo de imagen
        config: Diccionario de configuración
    
    Returns:
        Tensor [H, W, 3] en float32, rango [0, 255]
    """
    import tensorflow_io as tfio
    
    # Leer y decodificar TIFF
    image = tf.io.read_file(path)
    image = tfio.experimental.image.decode_tiff(image)
    
    # Manejar dimensión batch si existe
    if len(image.shape) == 4:
        image = tf.squeeze(image, axis=0)
    
    # Resize a resolución objetivo
    image = tf.image.resize(image, [config['img_height'], config['img_width']])
    
    # Tomar solo 3 canales (RGB)
    image = image[:, :, :3]
    
    # Convertir a float32
    image = tf.cast(image, tf.float32)
    
    return image


# =============================================================================
# SECCIÓN 2.5: PREPROCESSING ESPECÍFICO POR MODELO
# =============================================================================

def get_model_preprocessing(model_type: str):
    """
    Retorna la función de preprocessing específica para cada arquitectura.
    
    Cada modelo pre-entrenado requiere preprocessing diferente:
    - baseline: Rescale a [0, 1]
    - vgg: RGB→BGR + ImageNet mean subtraction
    - resnet: ImageNet mean/std normalization
    - efficientnet: Pass-through (modelo maneja internamente)
    - vit: ImageNet normalization (mean=0.5, std=0.5) -> [-1, 1]

    Args:
        model_type: 'baseline', 'vgg', 'resnet', 'efficientnet', 'vit'
    
    Returns:
        Función de preprocessing (image, label) -> (image, label)
    """
    model_type = model_type.lower()
    
    if model_type == 'baseline':
        def preprocess(image, label):
            # Simple rescale a [0, 1]
            return image / 255.0, label
        return preprocess
    
    elif model_type in ['vgg', 'vgg19', 'vgg16']:
        from tensorflow.keras.applications.vgg19 import preprocess_input
        def preprocess(image, label):
            # VGG: RGB→BGR + ImageNet mean subtraction
            return preprocess_input(image), label
        return preprocess
    
    elif model_type in ['resnet', 'resnet50']:
        from tensorflow.keras.applications.resnet import preprocess_input
        def preprocess(image, label):
            # ResNet: ImageNet mean/std normalization
            return preprocess_input(image), label
        return preprocess
    
    elif model_type in ['efficientnet', 'efficientnetb0']:
        from tensorflow.keras.applications.efficientnet import preprocess_input
        def preprocess(image, label):
            # EfficientNet: preprocess_input (pass-through en TF 2.10)
            return preprocess_input(image), label
        return preprocess

    elif model_type in ['vit', 'vit_b16', 'vit_s16']:
        def preprocess(image, label):
            # ViT: ImageNet normalization (mean=0.5, std=0.5) -> [-1, 1]
            # Rescale from [0, 255] to [0, 1], then normalize to [-1, 1]
            image = image / 255.0  # [0, 255] -> [0, 1]
            image = (image - 0.5) / 0.5  # [0, 1] -> [-1, 1]
            return image, label
        return preprocess

    else:
        raise ValueError(f"Modelo no soportado: {model_type}. "
                        f"Usar: 'baseline', 'vgg', 'resnet', 'efficientnet', 'vit'")


# =============================================================================
# SECCIÓN 2.6: CREACIÓN DE DATASETS
# =============================================================================

def create_dataset(
    directory: Path,
    config: Dict[str, Any],
    augment: bool = False,
    shuffle: bool = True
) -> tf.data.Dataset:
    """
    Crea un tf.data.Dataset completo desde un directorio.
    
    Pipeline (orden crítico):
    1. Load TIFF → [0, 255]
    2. Color Augmentation (solo train + baseline)
    3. Geometric Augmentation (solo train)
    4. Model-specific Preprocessing
    5. Batch + Prefetch
    
    Args:
        directory: Directorio con subdirectorios por clase
        config: Diccionario de configuración
        augment: Aplicar augmentation (True para train)
        shuffle: Mezclar dataset
    
    Returns:
        tf.data.Dataset configurado
    """
    # Recolectar rutas y etiquetas
    file_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = directory / class_name
        if class_dir.exists():
            class_files = list(class_dir.glob('*.tif'))
            file_paths.extend([str(f) for f in class_files])
            labels.extend([class_idx] * len(class_files))
    
    # Crear dataset base
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(file_paths), 
            seed=config['seed']
        )
    
    # =========================================================================
    # PIPELINE (orden según README §6.1)
    # =========================================================================
    
    # Paso 1: Cargar imagen
    dataset = dataset.map(
        lambda path, label: (load_image(path, config), label),
        num_parallel_calls=AUTOTUNE
    )
    
    # Establecer shapes explícitamente
    dataset = dataset.map(
        lambda x, y: (
            tf.ensure_shape(x, [config['img_height'], config['img_width'], 3]),
            tf.ensure_shape(y, [])
        )
    )
    
    # Paso 2: Color Augmentation (solo train + baseline)
    if augment and config['use_color_augmentation']:
        dataset = dataset.map(
            lambda x, y: (color_augmentation(x, config), y),
            num_parallel_calls=AUTOTUNE
        )
    
    # Paso 3: Geometric Augmentation (solo train)
    if augment:
        dataset = dataset.map(
            lambda x, y: (geometric_augmentation(x, config), y),
            num_parallel_calls=AUTOTUNE
        )
    
    # Paso 4: Model-specific Preprocessing
    preprocess_fn = get_model_preprocessing(config['model_type'])
    dataset = dataset.map(preprocess_fn, num_parallel_calls=AUTOTUNE)
    
    # Paso 5: Batch + Prefetch
    dataset = dataset.batch(config['batch_size'])
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset


def create_datasets(
    data_dir: str = '../data',
    model_type: str = 'baseline',
    img_size: int = 150,
    batch_size: int = 32,
    seed: int = 42,
    use_color_augmentation: bool = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Crea los tres datasets (train, val, test) con configuración óptima.
    
    FUNCIÓN PRINCIPAL - Usar esta para crear pipelines en notebooks.
    
    Args:
        data_dir: Directorio de datos (con subdirs train/, val/, test/)
        model_type: 'baseline', 'vgg', 'resnet', 'efficientnet'
        img_size: Tamaño de imagen (150 para baseline, 224 para TL)
        batch_size: Tamaño de batch
        seed: Semilla para reproducibilidad
        use_color_augmentation: Habilitar color aug (default: True solo para baseline)
    
    Returns:
        Tupla (train_dataset, val_dataset, test_dataset)
    
    Ejemplo:
        # CNN Baseline
        train_ds, val_ds, test_ds = create_datasets(
            model_type='baseline', img_size=150
        )
        
        # ResNet50
        train_ds, val_ds, test_ds = create_datasets(
            model_type='resnet', img_size=224
        )
    """
    # Generar configuración
    if use_color_augmentation is None:
        use_color_augmentation = (model_type.lower() == 'baseline')
    
    config = get_config(
        model_type=model_type,
        img_size=img_size,
        batch_size=batch_size,
        seed=seed,
        data_dir=data_dir,
        use_color_augmentation=use_color_augmentation
    )
    
    data_path = Path(data_dir)
    
    print(f"\n{'='*60}")
    print("CREANDO DATA PIPELINE")
    print(f"{'='*60}")
    print(f"Modelo: {config['model_type'].upper()}")
    print(f"Resolución: {config['img_height']}x{config['img_width']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Color augmentation: {'✓' if config['use_color_augmentation'] else '✗'}")
    print(f"Geometric augmentation: ✓ (estandarizado)")
    print(f"{'='*60}\n")
    
    # Training: con augmentation, con shuffle
    train_dataset = create_dataset(
        data_path / 'train',
        config,
        augment=True,
        shuffle=True
    )
    
    # Validation: sin augmentation, sin shuffle, con cache
    val_dataset = create_dataset(
        data_path / 'val',
        config,
        augment=False,
        shuffle=False
    )
    val_dataset = val_dataset.cache()
    
    # Test: sin augmentation, sin shuffle, con cache
    test_dataset = create_dataset(
        data_path / 'test',
        config,
        augment=False,
        shuffle=False
    )
    test_dataset = test_dataset.cache()
    
    # Contar samples
    n_train = sum(1 for _ in (data_path / 'train').rglob('*.tif'))
    n_val = sum(1 for _ in (data_path / 'val').rglob('*.tif'))
    n_test = sum(1 for _ in (data_path / 'test').rglob('*.tif'))
    
    print(f"Training samples: {n_train} ({n_train // config['batch_size']} batches)")
    print(f"Validation samples: {n_val} ({n_val // config['batch_size']} batches)")
    print(f"Test samples: {n_test} ({n_test // config['batch_size']} batches)")
    print(f"\n✓ Pipeline GPU-optimizado (sin py_function)")
    
    return train_dataset, val_dataset, test_dataset


# =============================================================================
# UTILIDADES
# =============================================================================

def visualize_batch(dataset: tf.data.Dataset, class_names: list = CLASS_NAMES, n_images: int = 16):
    """
    Visualiza un batch del dataset.
    
    Args:
        dataset: Dataset a visualizar
        class_names: Lista de nombres de clases
        n_images: Número de imágenes a mostrar
    """
    import matplotlib.pyplot as plt
    
    # Obtener un batch
    batch = next(iter(dataset))
    images, labels = batch
    
    # Desnormalizar si es necesario para visualización
    images_vis = images.numpy()
    if images_vis.min() < 0:  # Probablemente normalizado con ImageNet
        # Aproximación: escalar a [0, 1]
        images_vis = (images_vis - images_vis.min()) / (images_vis.max() - images_vis.min())
    elif images_vis.max() <= 1.0:
        pass  # Ya en [0, 1]
    else:
        images_vis = images_vis / 255.0
    
    # Plot
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols
    
    plt.figure(figsize=(12, 3 * n_rows))
    for i in range(min(n_images, len(images))):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(images_vis[i])
        plt.title(class_names[labels[i].numpy()])
        plt.axis('off')
    
    plt.suptitle('Sample Batch', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print(f"Batch shape: {images.shape}")
    print(f"Value range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("TEST: Shared Data Pipeline")
    print("="*60)
    
    # Test para cada modelo
    for model in ['baseline', 'vgg', 'resnet', 'efficientnet']:
        print(f"\n--- Testing {model.upper()} ---")
        
        img_size = 150 if model == 'baseline' else 224
        
        try:
            # API simplificada
            train_ds, val_ds, test_ds = create_datasets(
                data_dir='../data',
                model_type=model,
                img_size=img_size,
                batch_size=4  # Pequeño para test
            )
            
            # Verificar un batch
            batch = next(iter(train_ds))
            images, labels = batch
            print(f"  ✓ Shape: {images.shape}")
            print(f"  ✓ Range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETADO")
    print("="*60)
