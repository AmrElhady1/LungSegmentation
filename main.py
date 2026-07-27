import glob
import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D, Input, MaxPool2D, UpSampling2D)
from tensorflow.keras.models import Model
from tqdm import tqdm


def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def build_unet(input_shape):
    encoder_filters = [64, 128, 256, 512]
    decoder_filters = encoder_filters[::-1]

    inputs = Input(shape=input_shape)
    x = inputs
    skip_connections = []

    for f in encoder_filters:
        x = conv_block(x, f)
        skip_connections.append(x)
        x = MaxPool2D((2, 2))(x)

    x = conv_block(x, 1024)

    skip_connections = skip_connections[::-1]
    for i, f in enumerate(decoder_filters):
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, skip_connections[i]])
        x = conv_block(x, f)

    x = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=x)


def verify_file_list(file_list, description):
    if not file_list:
        raise FileNotFoundError(f"No files found for {description}")
    return sorted(file_list)


def preprocess_image(image_path, width, height):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image = cv2.resize(image, (width, height))
    image = image.astype(np.float32) / 255.0
    return image


def preprocess_mask(mask_path, width, height):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {mask_path}")
    mask = cv2.resize(mask, (width, height))
    mask = np.where(mask > 0, 1.0, 0.0).astype(np.float32)
    return mask


def load_dataset(image_paths, left_mask_paths, right_mask_paths, width, height):
    num_files = min(len(image_paths), len(left_mask_paths), len(right_mask_paths))
    if num_files == 0:
        raise ValueError("No matching image and mask files found.")

    if len(image_paths) != num_files or len(left_mask_paths) != num_files or len(right_mask_paths) != num_files:
        print(f"Warning: using first {num_files} matched entries")

    images = []
    masks = []

    for image_path, left_mask_path, right_mask_path in tqdm(
        zip(image_paths[:num_files], left_mask_paths[:num_files], right_mask_paths[:num_files]),
        total=num_files,
        desc="Loading dataset",
    ):
        images.append(preprocess_image(image_path, width, height))

        left_mask = preprocess_mask(left_mask_path, width, height)
        right_mask = preprocess_mask(right_mask_path, width, height)
        mask = np.clip(left_mask + right_mask, 0.0, 1.0)
        masks.append(mask)

    images = np.array(images, dtype=np.float32)
    masks = np.expand_dims(np.array(masks, dtype=np.float32), axis=-1)
    return images, masks


def save_numpy_arrays(output_dir, train_images, train_masks, valid_images, valid_masks):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "Unet-Train-Lung-Images.npy"), train_images)
    np.save(os.path.join(output_dir, "Unet-Train-Lung-Masks.npy"), train_masks)
    np.save(os.path.join(output_dir, "Unet-Validate-Lung-Images.npy"), valid_images)
    np.save(os.path.join(output_dir, "Unet-Validate-Lung-Masks.npy"), valid_masks)


def load_numpy_arrays(output_dir):
    train_images = np.load(os.path.join(output_dir, "Unet-Train-Lung-Images.npy"))
    train_masks = np.load(os.path.join(output_dir, "Unet-Train-Lung-Masks.npy"))
    valid_images = np.load(os.path.join(output_dir, "Unet-Validate-Lung-Images.npy"))
    valid_masks = np.load(os.path.join(output_dir, "Unet-Validate-Lung-Masks.npy"))
    return train_images, train_masks, valid_images, valid_masks


def predict_and_show(model, test_image_path, width, height):
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Test image not found at: {test_image_path}")
        return

    resized = cv2.resize(image, (width, height)).astype(np.float32) / 255.0
    prediction = model.predict(np.expand_dims(resized, axis=0))
    result_mask = (prediction[0] > 0.5).astype(np.uint8) * 255

    scale_percent = 60
    w = int(image.shape[1] * scale_percent / 100)
    h = int(image.shape[0] * scale_percent / 100)
    dim = (w, h)

    image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(result_mask, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow("Original Image", image_resized)
    cv2.imshow("Predicted Mask", mask_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    height = 256
    width = 256
    base_path = "D:/Data-sets/Lung Segmentation/MontgomerySet/"
    output_dir = "D:/temp"
    test_image_path = os.path.join(base_path, "Lung-test-Image-From-Montgomery.png")

    image_paths = verify_file_list(glob.glob(os.path.join(base_path, "CXR_png", "*.png")), "images")
    left_mask_paths = verify_file_list(glob.glob(os.path.join(base_path, "ManualMask", "LeftMask", "*.png")), "left masks")
    right_mask_paths = verify_file_list(glob.glob(os.path.join(base_path, "ManualMask", "rightMask", "*.png")), "right masks")

    images, masks = load_dataset(image_paths, left_mask_paths, right_mask_paths, width, height)
    print("Loaded dataset shapes:", images.shape, masks.shape)

    split = 0.1
    train_images, valid_images, train_masks, valid_masks = train_test_split(
        images, masks, test_size=split, random_state=42,
    )
    print("Train images:", train_images.shape)
    print("Train masks:", train_masks.shape)
    print("Valid images:", valid_images.shape)
    print("Valid masks:", valid_masks.shape)

    save_numpy_arrays(output_dir, train_images, train_masks, valid_images, valid_masks)
    print("Saved processed data to", output_dir)

    train_images, train_masks, valid_images, valid_masks = load_numpy_arrays(output_dir)

    model = build_unet((height, width, 3))
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    checkpoint_path = os.path.join(output_dir, "lung-Unet.h5")
    callbacks = [
        ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1, verbose=1, min_lr=1e-7),
        EarlyStopping(monitor="val_accuracy", patience=20, verbose=1),
    ]

    model.fit(
        train_images,
        train_masks,
        batch_size=4,
        epochs=50,
        verbose=1,
        validation_data=(valid_images, valid_masks),
        shuffle=True,
        callbacks=callbacks,
    )

    model = tf.keras.models.load_model(checkpoint_path)
    predict_and_show(model, test_image_path, width, height)


if __name__ == "__main__":
    main()
