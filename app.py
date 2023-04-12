import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# Load the model
def load_model(model_path):
    model_name = 'efficientnet-b0'
    model = EfficientNet.from_pretrained(model_name)
    num_features = model._fc.in_features
    model._fc = nn.Linear(num_features, 1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Image transformations
img_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

# Inference function
def predict_pawpularity(model, image):
    image_tensor = img_transforms(image)
    image_tensor = image_tensor.unsqueeze(0)
    output = model(image_tensor)
    return output.item()

# Load the trained model
model_path = 'trained_efficientnet_b0.pth'
model = load_model(model_path)

# Streamlit app
st.title("Pet Pawpularity Predictor")

uploaded_file = st.file_uploader("Upload an image of a pet", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded pet image", use_column_width=True)
    
    if st.button("Predict"):
        pawpularity_score = predict_pawpularity(model, image)
        st.write(f"The predicted pawpularity score is: {pawpularity_score:.2f}")
