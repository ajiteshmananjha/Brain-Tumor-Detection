import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the output size after 5 conv+pool layers for input 3x224x224
        # 224 -> 222 -> 111 -> 109 -> 54 -> 52 -> 26 -> 24 -> 12 -> 10 -> 5
        self.fc1 = nn.Linear(512 * 5 * 5, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn_fc3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
MODEL_PATH = "model_full.h5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
@st.cache_resource
def load_model():
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    return model.to(DEVICE)

model = load_model()

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Streamlit app ---
st.title("🧠 Brain Tumor Classifier (Custom CNN)")
st.markdown("Upload an MRI image to predict the tumor type.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)


    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_class = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx].item()

    st.markdown(f"### 🧠 Prediction: `{pred_class}`")
    st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")

    st.subheader("🔍 Class Probabilities:")
    for i, p in enumerate(probs):
        st.write(f"{CLASS_NAMES[i]}: {p.item() * 100:.2f}%")
