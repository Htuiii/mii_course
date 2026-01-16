# Установка необходимых библиотек для работы с NIfTI файлами
!pip install -q psutil kagglehub opencv-python scipy nibabel

import tensorflow as tf
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
from pathlib import Path
import os
import cv2
import nibabel as nib
from keras.saving import register_keras_serializable
# ============================================================================
# 1. КОНФИГУРАЦИЯ
# ============================================================================

class EfficientBRATSConfig:
    def __init__(self):
        self.IMG_SIZE = (240, 240)
        self.NUM_MODALITIES = 4
        self.NUM_CLASSES = 4

        self.BATCH_SIZE = 2
        self.VOLUME_DEPTH = 32
        self.EMBED_DIM = 128

        self.NUM_HEADS = 4
        self.NUM_TRANSFORMER_LAYERS = 8
        self.MLP_RATIO = 2
        self.CLASS_WEIGHTS = [0.1, 12.0, 8.0, 15.0]
        self.LEARNING_RATE = 1e-4
        self.EPOCHS = 50
        self.VALIDATION_SPLIT = 0.2

        self.MODELS_DIR = Path("/content/models/brats")
        self.VISUALIZATIONS_DIR = Path("/content/visualizations")
        for dir_path in [self.MODELS_DIR, self.VISUALIZATIONS_DIR]:
            dir_path.mkdir(exist_ok=True, parents=True)
    def get_config(self):
         return {
            'IMG_SIZE': self.IMG_SIZE,
            'NUM_MODALITIES': self.NUM_MODALITIES,
            'NUM_CLASSES': self.NUM_CLASSES,
            'BATCH_SIZE': self.BATCH_SIZE,
            'VOLUME_DEPTH': self.VOLUME_DEPTH,
            'EMBED_DIM': self.EMBED_DIM,
            'NUM_HEADS': self.NUM_HEADS,
            'NUM_TRANSFORMER_LAYERS': self.NUM_TRANSFORMER_LAYERS,
            'MLP_RATIO': self.MLP_RATIO,
            'CLASS_WEIGHTS': self.CLASS_WEIGHTS,
            'LEARNING_RATE': self.LEARNING_RATE,
            'EPOCHS': self.EPOCHS,
            'VALIDATION_SPLIT': self.VALIDATION_SPLIT
        }
    @classmethod
    def from_config(cls, config):
        return cls(**config)
# ============================================================================
# 2. БАЗОВЫЕ КЛАССЫ (которые были пропущены)
# ============================================================================

class NIFTIBRATSLoader:
    """Базовый загрузчик данных BRATS"""

    def __init__(self, config: EfficientBRATSConfig, min_tumor_ratio: float = 0.01):
        self.config = config
        self.min_tumor_ratio = min_tumor_ratio
        self.metadata = None

    def load_minimal_metadata(self, data_dir) -> pd.DataFrame:
        """Загрузка метаданных о пациентах"""
        print("🔍 Поиск файлов BRATS 2019...")

        records = []

        # Проверяем все подкаталоги
        for root, dirs, files in os.walk(data_dir):
            # Ищем файлы с BraTS в названии
            nii_files = [f for f in files if f.endswith('.nii') or f.endswith('.nii.gz')]

            if not nii_files:
                continue

            # Группируем файлы по пациентам
            patient_files = {}
            for file in nii_files:
                file_path = os.path.join(root, file)
                file_lower = file.lower()

                # Извлекаем ID пациента
                if 'brats' in file_lower:
                    parts = file.split('_')
                    for part in parts:
                        if 'brats' in part.lower() and len(part) > 5:
                            patient_id = part
                            break
                    else:
                        patient_id = file.split('.')[0]
                else:
                    patient_id = os.path.basename(root)

                if patient_id not in patient_files:
                    patient_files[patient_id] = {}

                # Определяем тип файла
                if 't1.' in file_lower and 't1ce' not in file_lower:
                    patient_files[patient_id]['t1'] = file_path
                elif 't1ce' in file_lower or 't1gd' in file_lower:
                    patient_files[patient_id]['t1ce'] = file_path
                elif 't2.' in file_lower:
                    patient_files[patient_id]['t2'] = file_path
                elif 'flair' in file_lower:
                    patient_files[patient_id]['flair'] = file_path
                elif 'seg' in file_lower:
                    patient_files[patient_id]['seg'] = file_path

            # Добавляем записи
            for patient_id, files in patient_files.items():
                if all(key in files for key in ['t1', 't1ce', 't2', 'flair', 'seg']):
                    records.append({
                        'patient_id': patient_id,
                        't1_path': files['t1'],
                        't1ce_path': files['t1ce'],
                        't2_path': files['t2'],
                        'flair_path': files['flair'],
                        'seg_path': files['seg'],
                        'has_tumor': True
                    })

        self.metadata = pd.DataFrame(records)
        print(f"✅ Найдено пациентов: {len(self.metadata)}")
        return self.metadata

# ============================================================================
# 3. ФУНКЦИИ ОБРАБОТКИ ДАННЫХ
# ============================================================================

def nifti_to_onehot_mask_3d(mask_data, target_size=(240, 240), num_slices=32):
    """Преобразование 3D маски в one-hot формат"""
    if len(mask_data.shape) == 4:
        mask_data = mask_data[:, :, :, 0]

    # Центральные срезы
    total_slices = mask_data.shape[2]
    if total_slices <= num_slices:
        selected_slices = list(range(total_slices))
        while len(selected_slices) < num_slices:
            selected_slices.append(selected_slices[-1])
    else:
        start = (total_slices - num_slices) // 2
        selected_slices = list(range(start, start + num_slices))

    onehot_3d = []
    for slice_idx in selected_slices:
        slice_data = mask_data[:, :, slice_idx]
        slice_resized = cv2.resize(slice_data, target_size, interpolation=cv2.INTER_NEAREST)

        onehot = np.zeros((target_size[0], target_size[1], 4), dtype=np.float32)
        onehot[:, :, 1] = (slice_resized == 1).astype(np.float32)  # NCR/NET
        onehot[:, :, 2] = (slice_resized == 2).astype(np.float32)  # ED
        onehot[:, :, 3] = (slice_resized == 4).astype(np.float32)  # ET

        tumor_mask = (slice_resized == 1) | (slice_resized == 2) | (slice_resized == 4)
        onehot[:, :, 0] = np.where(tumor_mask, 0.0, 0.1)  # Background

        onehot_3d.append(onehot)

    return np.stack(onehot_3d, axis=0)

def load_nifti_volume_3d(modality_files, target_size=(240, 240), num_slices=32):
    """Загрузка 3D объема"""
    volumes_3d = []

    for modality_file in modality_files:
        if not os.path.exists(modality_file):
            empty_volume = np.zeros((num_slices, target_size[0], target_size[1], 1), dtype=np.float32)
            volumes_3d.append(empty_volume)
            continue

        try:
            nifti_img = nib.load(modality_file)
            volume_data = nifti_img.get_fdata().astype(np.float32)

            # Нормализация
            non_zero = volume_data[volume_data > 0]
            if len(non_zero) > 10:
                p0_5 = np.percentile(non_zero, 0.5)
                p99_5 = np.percentile(non_zero, 99.5)
                volume_data = np.clip(volume_data, p0_5, p99_5)
                volume_data = (volume_data - p0_5) / (p99_5 - p0_5 + 1e-6)

            # Выбираем срезы
            total_slices = volume_data.shape[2]
            if total_slices <= num_slices:
                selected_slices = list(range(total_slices))
                while len(selected_slices) < num_slices:
                    selected_slices.append(selected_slices[-1])
            else:
                start = (total_slices - num_slices) // 2
                selected_slices = list(range(start, start + num_slices))

            # Ресайзим
            slices_resized = []
            for slice_idx in selected_slices:
                slice_2d = volume_data[:, :, slice_idx]
                slice_resized = cv2.resize(slice_2d, target_size, interpolation=cv2.INTER_LINEAR)
                slices_resized.append(slice_resized)

            volume_3d = np.stack(slices_resized, axis=0)
            volume_3d = np.expand_dims(volume_3d, axis=-1)
            volumes_3d.append(volume_3d)

        except Exception as e:
            empty_volume = np.zeros((num_slices, target_size[0], target_size[1], 1), dtype=np.float32)
            volumes_3d.append(empty_volume)

    if len(volumes_3d) == 4:
        return np.concatenate(volumes_3d, axis=-1)
    else:
        return np.zeros((num_slices, target_size[0], target_size[1], 4), dtype=np.float32)

# ============================================================================
# 4. УЛУЧШЕННЫЙ ЗАГРУЗЧИК
# ============================================================================

class ImprovedNIFTIBRATSLoader(NIFTIBRATSLoader):
    def __init__(self, config: EfficientBRATSConfig, min_tumor_ratio: float = 0.05):
        super().__init__(config, min_tumor_ratio)
        self.tumor_patients = []
        self.non_tumor_patients = []

    def load_minimal_metadata(self, data_dir) -> pd.DataFrame:
        """Переопределение метода с сохранением в self.metadata"""
        print("🔍 Поиск файлов BRATS 2019...")

        records = []

        # Проверяем все подкаталоги
        for root, dirs, files in os.walk(data_dir):
            # Ищем файлы с BraTS в названии
            nii_files = [f for f in files if f.endswith('.nii') or f.endswith('.nii.gz')]

            if not nii_files:
                continue

            # Группируем файлы по пациентам
            patient_files = {}
            for file in nii_files:
                file_path = os.path.join(root, file)
                file_lower = file.lower()

                # Извлекаем ID пациента
                if 'brats' in file_lower:
                    parts = file.split('_')
                    for part in parts:
                        if 'brats' in part.lower() and len(part) > 5:
                            patient_id = part
                            break
                    else:
                        patient_id = file.split('.')[0]
                else:
                    patient_id = os.path.basename(root)

                if patient_id not in patient_files:
                    patient_files[patient_id] = {}

                # Определяем тип файла
                if 't1.' in file_lower and 't1ce' not in file_lower:
                    patient_files[patient_id]['t1'] = file_path
                elif 't1ce' in file_lower or 't1gd' in file_lower:
                    patient_files[patient_id]['t1ce'] = file_path
                elif 't2.' in file_lower:
                    patient_files[patient_id]['t2'] = file_path
                elif 'flair' in file_lower:
                    patient_files[patient_id]['flair'] = file_path
                elif 'seg' in file_lower:
                    patient_files[patient_id]['seg'] = file_path

            # Добавляем записи
            for patient_id, files in patient_files.items():
                if all(key in files for key in ['t1', 't1ce', 't2', 'flair', 'seg']):
                    records.append({
                        'patient_id': patient_id,
                        't1_path': files['t1'],
                        't1ce_path': files['t1ce'],
                        't2_path': files['t2'],
                        'flair_path': files['flair'],
                        'seg_path': files['seg'],
                        'has_tumor': True
                    })

        self.metadata = pd.DataFrame(records)  # Важно сохранить в self.metadata
        print(f"✅ Найдено пациентов: {len(self.metadata)}")

        if len(self.metadata) > 0:
            # Для простоты считаем всех с опухолями
            self.tumor_patients = self.metadata['patient_id'].tolist()
            print(f"Пациентов с опухолями: {len(self.tumor_patients)}")
            print(f"Пациентов без опухолей: 0")

        return self.metadata

    def load_single_patient_3d(self, patient_id: str, require_tumor: bool = False):
        """Загрузка 3D данных одного пациента"""
        if self.metadata is None or len(self.metadata) == 0:
            return None, None

        patient_data = self.metadata[self.metadata['patient_id'] == patient_id]
        if len(patient_data) == 0:
            return None, None

        patient_row = patient_data.iloc[0]

        try:
            # Загружаем 4 модальности
            modality_files = [
                patient_row['t1_path'],
                patient_row['t1ce_path'],
                patient_row['t2_path'],
                patient_row['flair_path']
            ]

            volume = load_nifti_volume_3d(
                modality_files,
                target_size=self.config.IMG_SIZE,
                num_slices=self.config.VOLUME_DEPTH
            )

            # Загружаем маску
            seg_path = patient_row['seg_path']
            nifti_img = nib.load(seg_path)
            mask_data = nifti_img.get_fdata()

            mask = nifti_to_onehot_mask_3d(
                mask_data,
                target_size=self.config.IMG_SIZE,
                num_slices=self.config.VOLUME_DEPTH
            )

            if volume.shape != mask.shape:
                return None, None

            # Анализ опухоли
            tumor_pixels = np.sum(np.argmax(mask, axis=-1) > 0)
            tumor_percent = tumor_pixels / (mask.shape[0] * mask.shape[1] * mask.shape[2]) * 100

            if tumor_percent > 1:
                print(f"  {patient_id}: {volume.shape}, опухоль {tumor_percent:.2f}%")
            else:
                print(f"  {patient_id}: {volume.shape}, без опухоли")

            return volume, mask

        except Exception as e:
            print(f"  Ошибка загрузки {patient_id}: {e}")
            return None, None

# ============================================================================
# 5. ФУНКЦИИ ПОТЕРЬ
# ============================================================================

def hybrid_dice_loss(y_true, y_pred, config):
    """Упрощенная функция потерь"""
    class_weights = tf.constant(config.CLASS_WEIGHTS, dtype=tf.float32)

    # Weighted Dice Loss
    dice_losses = []
    for i in range(config.NUM_CLASSES):
        intersection = tf.reduce_sum(y_true[..., i] * y_pred[..., i])
        union = tf.reduce_sum(y_true[..., i]) + tf.reduce_sum(y_pred[..., i])
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_losses.append(class_weights[i] * (1 - dice))

    weighted_dice = tf.reduce_sum(dice_losses) / tf.reduce_sum(class_weights)

    # Binary crossentropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    cfe = tf.keras.losses.CategoricalFocalCrossentropy()(y_true, y_pred)
    tversky = tf.keras.losses.tversky(y_true, y_pred, alpha=0.7, beta=0.3, axis=None)

    return 0.2 * weighted_dice + 0.2 * bce + 0.5*tversky +0.5*cfe
@register_keras_serializable()
class HybridDiceLoss(tf.keras.losses.Loss):
    """Сериализуемая версия hybrid_dice_loss"""
    def __init__(self, config=None, name="hybrid_dice_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = config

    def call(self, y_true, y_pred):
        if self.config is None:
            # Создаем дефолтный конфиг, если не передан
            self.config = EfficientBRATSConfig()
        return hybrid_dice_loss(y_true, y_pred, self.config)

    def get_config(self):
        config = super().get_config()
        if self.config:
            config.update({'config': self.config.get_config()})
        return config

    @classmethod
    def from_config(cls, config):
        if 'config' in config:
            # Восстанавливаем конфиг из словаря
            cfg_obj = EfficientBRATSConfig()
            cfg_dict = config.pop('config')
            for key, value in cfg_dict.items():
                if hasattr(cfg_obj, key):
                    setattr(cfg_obj, key, value)
            config['config'] = cfg_obj
        return cls(**config)
# ============================================================================
# 6. АРХИТЕКТУРА МОДЕЛИ
# ============================================================================
@register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_ratio=2, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate
        )

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(embed_dim * mlp_ratio, activation='gelu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(embed_dim),
            tf.keras.layers.Dropout(dropout_rate)
        ])

    def call(self, inputs, training=False):
        x = self.norm1(inputs)
        attn_output = self.attn(x, x, training=training)
        x = inputs + attn_output

        y = self.norm2(x)
        mlp_output = self.mlp(y, training=training)
        x = x + mlp_output

        return x
    def get_config(self):  # ДОБАВЬТЕ ЭТОТ МЕТОД!
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'dropout_rate': self.dropout_rate
        })
        return config

def create_fixed_transformer_model(config: EfficientBRATSConfig):
    """Создание Transformer U-Net модели"""
    input_shape = (config.VOLUME_DEPTH, *config.IMG_SIZE, config.NUM_MODALITIES)
    inputs = tf.keras.Input(shape=input_shape)

    print("="*60)
    print("АРХИТЕКТУРА МОДЕЛИ")
    print(f"Вход: {input_shape}")
    print("="*60)

    # Энкодер
    x1 = tf.keras.layers.Conv3D(16, 3, padding='same')(inputs)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.ReLU()(x1)
    x1 = tf.keras.layers.Conv3D(16, 3, padding='same')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.ReLU()(x1)
    skip1 = x1
    x1_pool = tf.keras.layers.MaxPool3D(2)(x1)

    x2 = tf.keras.layers.Conv3D(32, 3, padding='same')(x1_pool)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)
    x2 = tf.keras.layers.Conv3D(32, 3, padding='same')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)
    skip2 = x2
    x2_pool = tf.keras.layers.MaxPool3D(2)(x2)

    x3 = tf.keras.layers.Conv3D(64, 3, padding='same')(x2_pool)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.ReLU()(x3)
    x3 = tf.keras.layers.Conv3D(64, 3, padding='same')(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.ReLU()(x3)
    skip3 = x3
    x3_pool = tf.keras.layers.MaxPool3D(2)(x3)

    x4 = tf.keras.layers.Conv3D(128, 3, padding='same')(x3_pool)
    x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.ReLU()(x4)
    x4 = tf.keras.layers.Conv3D(128, 3, padding='same')(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.ReLU()(x4)


    # Декодер
    up0 = tf.keras.layers.Conv3DTranspose(64, 2, strides=2, padding='same')(x4)
    concat0 = tf.keras.layers.Concatenate()([up0, skip3])
    dec0 = tf.keras.layers.Conv3D(64, 3, padding='same')(concat0)
    dec0 = tf.keras.layers.BatchNormalization()(dec0)
    dec0 = tf.keras.layers.ReLU()(dec0)
    dec0 = tf.keras.layers.Conv3D(64, 3, padding='same')(dec0)
    dec0 = tf.keras.layers.BatchNormalization()(dec0)
    dec0 = tf.keras.layers.ReLU()(dec0)

    up1 = tf.keras.layers.Conv3DTranspose(32, 2, strides=2, padding='same')(dec0)
    concat1 = tf.keras.layers.Concatenate()([up1, skip2])
    dec1 = tf.keras.layers.Conv3D(32, 3, padding='same')(concat1)
    dec1 = tf.keras.layers.BatchNormalization()(dec1)
    dec1 = tf.keras.layers.ReLU()(dec1)
    dec1 = tf.keras.layers.Conv3D(32, 3, padding='same')(dec1)
    dec1 = tf.keras.layers.BatchNormalization()(dec1)
    dec1 = tf.keras.layers.ReLU()(dec1)

    up2 = tf.keras.layers.Conv3DTranspose(16, 2, strides=2, padding='same')(dec1)
    concat2 = tf.keras.layers.Concatenate()([up2, skip1])
    dec2 = tf.keras.layers.Conv3D(16, 3, padding='same')(concat2)
    dec2 = tf.keras.layers.BatchNormalization()(dec2)
    dec2 = tf.keras.layers.ReLU()(dec2)
    dec2 = tf.keras.layers.Conv3D(16, 3, padding='same')(dec2)
    dec2 = tf.keras.layers.BatchNormalization()(dec2)
    dec2 = tf.keras.layers.ReLU()(dec2)

    # Выходной слой
    outputs = tf.keras.layers.Conv3D(config.NUM_CLASSES, 1, activation='softmax')(dec2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Fixed_Transformer_UNET")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=HybridDiceLoss(config),
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )

    return model

# ============================================================================
# 7. ТРЕНЕР
# ============================================================================

class ImprovedBalancedTrainer:
    def __init__(self, config: EfficientBRATSConfig):
        self.config = config
        self.data_loader = ImprovedNIFTIBRATSLoader(config, min_tumor_ratio=0.05)
        self.model = None

    def create_balanced_dataset(self, data_dir, n_patients: int = 30):
        """Создание датасета"""
        print("\n📊 СОЗДАНИЕ ДАТАСЕТА...")

        # ИСПРАВЛЕНИЕ ЗДЕСЬ: используем data_loader напрямую
        df = self.data_loader.load_minimal_metadata(data_dir)

        if df is None or len(df) == 0:
            print("❌ Нет данных!")
            return [], []

        print(f"✅ Найдено пациентов: {len(df)}")

        # Выбор пациентов
        if n_patients is None or n_patients > len(df):
            selected_patients = df['patient_id'].tolist()  # Все пациенты
        else:
            selected_patients = df['patient_id'].tolist()[:n_patients]

        print(f"Выбрано {len(selected_patients)} пациентов")

        # Разделение
        split_idx = int(len(selected_patients) * (1 - self.config.VALIDATION_SPLIT))
        train_ids = selected_patients[:split_idx]
        val_ids = selected_patients[split_idx:]

        print(f"  Train: {len(train_ids)} пациентов")
        print(f"  Val: {len(val_ids)} пациентов")

        return train_ids, val_ids

    def train_balanced(self, data_dir, n_patients: int = 40):
        """Обучение модели"""
        print("="*70)
        print("ОБУЧЕНИЕ МОДЕЛИ")
        print("="*70)

        # 1. Создание датасета
        train_ids, val_ids = self.create_balanced_dataset(data_dir, n_patients)

        if len(train_ids) < 3:
            print("❌ Недостаточно данных!")
            return None, {}, train_ids, val_ids

        # 2. Создание модели
        print("\n🤖 СОЗДАНИЕ МОДЕЛИ...")
        self.model = create_fixed_transformer_model(self.config)

        # 3. Загрузка данных
        train_data = []
        train_labels = []

        # Загрузка тренировочных данных
        successful = 0
        for pid in train_ids:
            data = self.data_loader.load_single_patient_3d(pid, require_tumor=False)
            if data[0] is not None:
                volume, mask = data
                train_data.append(volume)
                train_labels.append(mask)
                successful += 1
                del volume, mask
                gc.collect()

        print(f"✅ Успешно загружено: {successful} из {len(train_ids)} тренировочных пациентов")

        if successful < 3:
            print("❌ Недостаточно данных!")
            return None, {}, train_ids, val_ids

        train_data = np.array(train_data)
        train_labels = np.array(train_labels)

        print(f"✅ Данные загружены: {train_data.shape}")

        # 4. Обучение
        print("\n🚀 НАЧАЛО ОБУЧЕНИЯ...")

        history = self.model.fit(
            train_data, train_labels,
            validation_split=self.config.VALIDATION_SPLIT,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
            ]
        )

        # 5. Сохранение
        model_path = self.config.MODELS_DIR / 'brain_tumor_model.keras'
        self.model.save(model_path)
        print(f"\n💾 Модель сохранена: {model_path}")

        return self.model, history.history, train_ids, val_ids

# ============================================================================
# 8. ВИЗУАЛИЗАЦИЯ
# ============================================================================

def visualize_training(history):
    """Визуализация процесса обучения"""
    if not history:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history['loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Recall
    axes[2].plot(history['recall'], 'g-', label='Train Recall', linewidth=2)
    axes[2].plot(history['val_recall'], 'orange', label='Val Recall', linewidth=2)
    axes[2].set_title('Recall', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Recall')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('ДИНАМИКА ОБУЧЕНИЯ', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()

def visualize_segmentation(model, data_loader, config, patient_id=None):
    """Визуализация сегментации"""
    print("\n" + "="*70)
    print("ВИЗУАЛИЗАЦИЯ СЕГМЕНТАЦИИ")
    print("="*70)

    if patient_id is None:
        if data_loader.metadata is not None and len(data_loader.metadata) > 0:
            patient_id = data_loader.metadata.iloc[0]['patient_id']
        else:
            print("❌ Нет данных для визуализации")
            return

    # Загружаем данные пациента
    volume, mask = data_loader.load_single_patient_3d(patient_id, require_tumor=False)

    if volume is None:
        print(f"❌ Не удалось загрузить пациента {patient_id}")
        return

    # Предсказание
    pred = model.predict(volume[np.newaxis, ...])[0]

    # Визуализация
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    slices = [8, 16, 24]  # Центральные срезы

    for i, slice_idx in enumerate(slices):
        # Исходное изображение (FLAIR)
        axes[i, 0].imshow(volume[slice_idx, :, :, 3], cmap='gray')
        axes[i, 0].set_title(f'МРТ - Срез {slice_idx}', fontweight='bold')
        axes[i, 0].axis('off')

        # Истинная сегментация
        true_seg = np.argmax(mask[slice_idx], axis=-1)
        axes[i, 1].imshow(true_seg, cmap='tab10', vmin=0, vmax=3)
        axes[i, 1].set_title('Истинная маска', fontweight='bold')
        axes[i, 1].axis('off')

        # Предсказанная сегментация
        pred_seg = np.argmax(pred[slice_idx], axis=-1)
        axes[i, 2].imshow(pred_seg, cmap='tab10', vmin=0, vmax=3)
        axes[i, 2].set_title('Предсказанная маска', fontweight='bold')
        axes[i, 2].axis('off')

        # Наложение
        overlay = volume[slice_idx, :, :, 3].copy()
        overlay = np.stack([overlay, overlay, overlay], axis=-1)

        # Цвета для классов опухоли
        colors = {
            1: [1, 0.5, 0],  # Красный для NCR/NET
            2: [0, 1, 0],  # Зеленый для ED
            3: [1, 0, 1]   # Синий для ET
        }

        for class_id, color in colors.items():
            tumor_mask = pred_seg == class_id
            overlay[tumor_mask] = color

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Наложение опухоли', fontweight='bold')
        axes[i, 3].axis('off')

    plt.suptitle(f'СЕГМЕНТАЦИЯ ОПУХОЛИ МОЗГА\nПациент: {patient_id}',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    # Статистика
    print("\n📊 СТАТИСТИКА СЕГМЕНТАЦИИ:")
    print("-" * 40)

    true_flat = np.argmax(mask, axis=-1).flatten()
    pred_flat = np.argmax(pred, axis=-1).flatten()

    classes = ['Фон', 'NCR/NET', 'ED', 'ET']

    for class_id in range(4):
        true_count = (true_flat == class_id).sum()
        pred_count = (pred_flat == class_id).sum()

        if class_id > 0:  # Для опухолевых классов считаем Dice
            intersection = ((true_flat == class_id) & (pred_flat == class_id)).sum()
            dice = 2 * intersection / (true_count + pred_count + 1e-6)
            dice_str = f", Dice: {dice:.3f}"
        else:
            dice_str = ""

        print(f"{classes[class_id]}: True={true_count:,}, Pred={pred_count:,}{dice_str}")

# ============================================================================
# 9. ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("="*80)
    print("СЕГМЕНТАЦИЯ ОПУХОЛЕЙ МОЗГА - TRANSFORMER 3D U-NET")
    print("="*80)

    # Инициализация
    config = EfficientBRATSConfig()

    # Проверяем данные
    print("\n1. ПРОВЕРКА ДАННЫХ...")

    data_path = "/kaggle/input/brain-tumor-segmentation-brats-2019"
    if not os.path.exists(data_path):
        try:
            import kagglehub
            data_path = kagglehub.dataset_download("aryashah2k/brain-tumor-segmentation-brats-2019")
            print(f"✅ Данные скачаны: {data_path}")
        except:
            print("❌ Не удалось найти данные")
            return

    # Обучение
    print(f"\n2. ОБУЧЕНИЕ МОДЕЛИ...")
    trainer = ImprovedBalancedTrainer(config)
    model, history, train_ids, val_ids = trainer.train_balanced(data_path, n_patients=40)

    if model is None:
        print("❌ Обучение не удалось")
        return

    # Визуализация
    print(f"\n3. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ...")

    # Графики обучения
    visualize_training(history)

    # Пример сегментации
    if train_ids:
        visualize_segmentation(model, trainer.data_loader, config, patient_id=train_ids[0])

    print("\n" + "="*80)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print("="*80)

# ============================================================================
# 10. ЗАПУСК В COLAB
# ============================================================================

if __name__ == "__main__":
    # Очистка памяти
    tf.keras.backend.clear_session()
    gc.collect()

    # Проверка GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU доступен")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️ Обучение на CPU")

    # Запуск
    main()
