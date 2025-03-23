# Object Classification Model Pipeline

# EfficientNetB0 Transfer Learning (TensorFlow/Keras)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_efficientnet_model(input_shape=(224, 224, 3), num_classes=256):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# SVM with Bag of Visual Words using OpenCV (SURF/SIFT)
def extract_sift_features(image_paths):
    import cv2
    sift = cv2.SIFT_create()
    descriptors = []
    for path in image_paths:
        img = cv2.imread(path, 0)
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            descriptors.append(des)
    return descriptors

# Bernoulli Naive Bayes classifier (baseline)
def train_bernoulli_nb(X_train, y_train):
    from sklearn.naive_bayes import BernoulliNB
    model = BernoulliNB()
    model.fit(X_train, y_train)
    return model
