# Shared Data Pipeline

## Descripción

El módulo `shared_pipeline.py` implementa un pipeline de datos estandarizado y GPU-optimizado para todos los modelos del proyecto.

## Características

- **100% TensorFlow nativo** - Sin `py_function`, máximo rendimiento GPU
- **Augmentation estandarizada** - Comparación justa entre modelos
- **Preprocessing específico** - Cada arquitectura recibe el formato correcto

## Uso Rápido

### En notebooks

```python
# Importar
import sys
sys.path.append(str(Path.cwd().parent / 'src'))
from shared_pipeline import create_datasets, CLASS_NAMES

# CNN Baseline (150×150, con color augmentation)
train_ds, val_ds, test_ds = create_datasets(
    data_dir='../data',
    model_type='baseline',
    img_size=150,
    batch_size=32
)

# VGG19 (224×224, VGG preprocessing)
train_ds, val_ds, test_ds = create_datasets(
    data_dir='../data',
    model_type='vgg',
    img_size=224,
    batch_size=32
)

# ResNet50 (224×224, ResNet preprocessing)
train_ds, val_ds, test_ds = create_datasets(
    data_dir='../data',
    model_type='resnet',
    img_size=224,
    batch_size=32
)

# EfficientNet (224×224, EfficientNet preprocessing)
train_ds, val_ds, test_ds = create_datasets(
    data_dir='../data',
    model_type='efficientnet',
    img_size=224,
    batch_size=32
)
```

## Pipeline por Modelo

| Modelo | Resolución | Color Aug | Preprocessing |
|--------|------------|-----------|---------------|
| baseline | 150×150 | ✓ HSV | `/ 255.0` → [0,1] |
| vgg | 224×224 | ✗ | RGB→BGR + mean sub |
| resnet | 224×224 | ✗ | mean/std norm |
| efficientnet | 224×224 | ✗ | pass-through |

## Orden del Pipeline

```
1. Load TIFF (tensorflow-io)
2. Resize (img_size × img_size)
3. Color Augmentation (solo train + baseline)
4. Geometric Augmentation (solo train)
   - Rotaciones 90° (k=0,1,2,3)
   - Flip horizontal/vertical
   - Zoom IN [1.0, 1.1]
5. Model-specific Preprocessing
6. Batch + Prefetch
```

## Justificación Académica

La estandarización del pipeline garantiza:

1. **Reproducibilidad** - Mismos datos para todos los modelos
2. **Comparación justa** - Diferencias de rendimiento se deben al modelo
3. **Práctica estándar** - Alineado con benchmarks de ImageNet/CIFAR

> "For fair comparison, all models should receive identical input data, with only model-specific preprocessing applied as the final step."
> — Raghu et al. (2019), "Transfusion: Understanding Transfer Learning for Medical Imaging"

## Archivos

- `shared_pipeline.py` - Módulo principal
- `data_pipeline.py` - Versión anterior (más compleja, con Macenko)
