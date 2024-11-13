import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# 설정
image_x, image_y = 224, 224  # ResNet 모델은 224x224 입력을 기대합니다
batch_size = 64
train_dir = "chords"  # 데이터셋 폴더 경로
num_of_classes = 7
num_epochs = 5
learning_rate = 0.001

# 데이터 전처리
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # 이미지 픽셀 값을 0~1로 정규화
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% 데이터는 검증용으로 분리
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_x, image_y),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"  # 학습 데이터
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_x, image_y),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"  # 검증 데이터
)

# ResNet 모델 로드 및 수정
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(image_x, image_y, 3))
base_model.trainable = False  # 사전 학습된 레이어 고정

# 새로운 분류기 추가
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),  # 드롭아웃 추가로 과적합 방지
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_of_classes, activation='softmax')  # 클래스 수에 맞게 출력층 수정
])

# 모델 컴파일
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 모델 학습
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator
)

# 학습 결과 저장
model.save("guitar_learner_resnet_tf.h5")

# 평가
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
