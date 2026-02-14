import tensorflow as tf
import cv2
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.
    
    Args:
        frame: Image that needs to resized and padded. 
        output_size: Pixel size of the output frame image.

    Return:
        Formatted frame with padding of specified output size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def frames_from_video_file(video_path, n_frames, output_size=(224,224), frame_step=15):
    """
    Creates frames from each video file present for each category.

    Args:
        video_path: File path to the video.
        n_frames: Number of frames to be created per video file.
        output_size: Pixel size of the output frame image.

    Return:
        An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))  

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = int(video_length - need_length)
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result

class FrameGenerator:
    def __init__(self, path, n_frames, training=False):
        """ Returns a set of frames with their associated label. 

        Args:
            path: Video file paths.
            n_frames: Number of frames. 
            training: Boolean to determine if training dataset is being created.
        """
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*.avi'))
        classes = [p.parent.name for p in video_paths] 
        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames) 
            label = self.class_ids_for_name[name] # Encode labels
            yield video_frames, label

class DepthwiseConv3D(tf.keras.layers.Layer):
    def __init__(self, kernel_size=(3, 3, 3), padding='same', **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.padding = padding

    def build(self, input_shape):
        # input_shape: (batch, time, height, width, channels)
        self.channels = input_shape[-1]
        self.depthwise_convs = [
            tf.keras.layers.Conv3D(
                filters=1,
                kernel_size=self.kernel_size,
                padding=self.padding,
                use_bias=False,
                groups=1  # emulate grouped conv per channel
            )
            for _ in range(self.channels)
        ]
        super().build(input_shape)

    def call(self, x):
        # x: (batch, time, height, width, channels)
        channels = tf.split(x, num_or_size_splits=self.channels, axis=-1)
        outputs = [conv(c) for conv, c in zip(self.depthwise_convs, channels)]
        return tf.concat(outputs, axis=-1)

def depthwise_separable_conv3d_block(input_tensor, filters, kernel_size=(3, 3, 3), padding='same'):
    # Step 1: Depthwise 3D convolution
    x = DepthwiseConv3D(kernel_size=kernel_size, padding=padding)(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Step 2: Pointwise Conv3D (1x1x1) to mix channels
    x = tf.keras.layers.Conv3D(filters, kernel_size=(1, 1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def build_true_csn_model(input_shape=(16, 224, 224, 3), num_classes=9):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # === Depthwise Conv Layers (no mixing) ===
    x = DepthwiseConv3D(kernel_size=(3, 3, 3))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = DepthwiseConv3D(kernel_size=(3, 3, 3))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = DepthwiseConv3D(kernel_size=(3, 3, 3))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # === Final Pointwise Conv for Channel Mixing ===
    x = tf.keras.layers.Conv3D(filters=256, kernel_size=(1, 1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # === Global Pooling and Classification ===
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

def get_train_val_test(n_frames=10, batch_size=2):
    output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                       tf.TensorSpec(shape=(), dtype=tf.int16))

    train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], 16, training=True),
                                            output_signature=output_signature)
    val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], 16),
                                          output_signature=output_signature)
    test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], 16),
                                           output_signature=output_signature)

    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    train_ds = train_ds.batch(2)
    val_ds = val_ds.batch(2)
    test_ds = test_ds.batch(2)

    return train_ds, val_ds, test_ds

# Create and compile the model
model = build_true_csn_model()
model.summary()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks to control the learning rate & early stopping
reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',    # or 'loss' if no validation set
    factor=0.5,            # reduce LR by a factor of 0.5
    patience=2,            # wait 2 epochs with no improvement
    verbose=1,             # print updates
    min_lr=1e-6            # never go below this
)

def scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch > 0:
        return lr * 0.5
    return lr

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

initial_lr = 1e-3
exp_decay = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_lr,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

early_stopping = tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')
# Add ModelCheckpoint callback to save model after each epoch
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5',
    monitor='val_loss',
    save_best_only=False,
    save_weights_only=False,
    verbose=1
)

# Load UCF101 subset
DATASET_PATH = "/Users/blag/Downloads/Computer Vision with DL 2/UCF101_subset"
SELECTED_CLASSES = [
    "ApplyEyeMakeup", "PlayingDhol", "BabyCrawling",
    "Haircut", "SkyDiving", "Surfing",
    "Rafting", "CricketShot", "ShavingBeard"
]

# Define the paths for train, validation, and test sets
subset_paths = {
    'train': Path(DATASET_PATH) / 'train',
    'val': Path(DATASET_PATH) / 'val',
    'test': Path(DATASET_PATH) / 'test'
}

# Get the datasets
train_ds, val_ds, test_ds = get_train_val_test()

# Train the model
history = model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds,
        callbacks=[reduce_lr_on_plateau, early_stopping, model_checkpoint]
)
model.save('newest_separable_csn_model.h5')

# Plot training history
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('training_history_separable.png')
plt.close()

# Evaluate the model on test set
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"\nTest accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Get predictions for confusion matrix
y_pred = []
y_true = []

for x, y in test_ds:
    predictions = model.predict(x)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(y.numpy())

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=SELECTED_CLASSES,
            yticklabels=SELECTED_CLASSES)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confupsion_matrix_separable.png')
plt.close()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=SELECTED_CLASSES)) 