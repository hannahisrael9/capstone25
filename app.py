import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import gradio as gr
from fpdf import FPDF
from torch import nn
import torchvision.models as models
import pandas as pd
import os


class VisionTransformerWithMetadata(nn.Module):
    def __init__(self, num_classes=8, metadata_features=3):
        super(VisionTransformerWithMetadata, self).__init__()
        # Vision Transformer (ViT) part - Basic architecture
        vit_weights_path = "/Users/hannah/.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth"
        self.vit = models.vit_b_16()
        self.vit.load_state_dict(torch.load(vit_weights_path, map_location=torch.device('cpu')))
        vit_in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(vit_in_features, num_classes)

        # EfficientNet
        efficientnet_weights_path = "/Users/hannah/.cache/torch/hub/checkpoints/efficientnet_b0.pth"
        self.efficientnet = models.efficientnet_b0()
        self.efficientnet.load_state_dict(torch.load(efficientnet_weights_path))
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)  # Modify for number of classes

        # Fully connected layer for metadata processing
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_features, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )

        # Fully connected layer to combine image + metadata features
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_classes * 2 + 32, num_classes)  # Combine ViT, EfficientNet & metadata
        )

    def forward(self, x, metadata):
        vit_features = self.vit(x)
        efficientnet_features = self.efficientnet(x)

        metadata_features = self.metadata_fc(metadata)  # Process metadata separately

        combined = torch.cat((vit_features, efficientnet_features, metadata_features), dim=1)
        out = self.fc(combined)
        return out

# Initialize the model
num_classes = 8  # Set this to the correct number of classes
metadata_features = 3  # Number of metadata features
model = VisionTransformerWithMetadata(num_classes=num_classes, metadata_features=metadata_features)

# Load the model state dictionary
model.load_state_dict(torch.load('pad-ham_model_metadata.pth'))

# Set the model to evaluation mode
model.eval()


# Define class labels and descriptions
skin_lesion_info = {
    "Melanoma": "Melanoma is the most serious type of skin cancer. Early detection is crucial as it can spread rapidly. Seek medical attention if you notice irregularly shaped or dark lesions.",
    "Nevus": "Nevus (mole) is a common, usually harmless growth of pigmented skin cells. However, changes in size, color, or shape could indicate potential malignancy.",
    "Basal Cell Carcinoma": "Basal Cell Carcinoma (BCC) is a slow-growing skin cancer that rarely spreads. It often appears as a shiny, pink, or pearly bump. Consult a dermatologist for further evaluation.",
    "Actinic Keratosis": "Actinic Keratosis (AK) is a precancerous lesion caused by sun exposure. It may appear as rough, scaly patches. Treatment is recommended to prevent progression to skin cancer.",
    "Benign Keratosis": "Benign Keratosis (BKL) includes solar lentigines, seborrheic keratoses, and lichen-planus-like keratoses. These are non-cancerous and require no treatment unless they change appearance.",
    "Dermatofibroma": "Dermatofibroma (DF) is a common benign skin nodule. It is firm to touch and typically harmless. No treatment is necessary unless for cosmetic reasons.",
    "Vascular Lesions": "Vascular Lesions (VASC) include angiomas and other blood vessel growths. These are typically benign but may be removed for cosmetic reasons.",
    "Squamous Cell Carcinoma": "Squamous Cell Carcinoma (SCC) is a common skin cancer that can spread if untreated. It appears as scaly, red patches or open sores. Medical evaluation is advised."
}

tensor_regions_cat = {
    'back': 0.15789474, 'lower extremity':  0.63157895, 'trunk': 0.89473684, 'face': 0.31578947, 'upper extremity': 1.0,
    'abdomen': 0.0, 'chest': 0.21052632, 'forearm': 0.42105263, 'foot': 0.36842105, 'neck': 0.68421053, 'unknown': 0.94736842,
    'hand': 0.52631579, 'arm': 0.10526316, 'nose': 0.73684211, 'scalp': 0.78947368, 'ear': 0.26315789, 'thigh': 0.84210526, 'genital': 0.47368421,
    'lip': 0.57894737, 'acral': 0.05263158
}

scaler = MinMaxScaler()

# Define the classify_image function
def classify_image(image_path, age, gender, region):
    """Classify a given image and return predictions with confidence scores."""
    # Preprocess Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Preprocess Metadata
    age = pd.to_numeric(age)  # Convert age to numeric

    gender_mapping = {'female': 0, 'male': 1, 'unknown': 2}
    gender_encoded = gender_mapping[gender.lower()]  

    region_mapping = tensor_regions_cat
    if region.lower() not in region_mapping:
        region_encoded = region_mapping['unknown']
    else:
        region_encoded = region_mapping[region.lower()]  # Encode lesion location


    # Normalize metadata values
    metadata_input = torch.tensor([[age, gender_encoded, region_encoded]], dtype=torch.float32)
    metadata_input = scaler.fit_transform(metadata_input)
    metadata_input = torch.tensor(metadata_input, dtype=torch.float32)  # Convert to tensor
    
    # Move inputs to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    metadata_input = metadata_input.to(device)
    model.to(device)

    # Make predictions
    with torch.no_grad():
        predictions = model(image, metadata_input)
    
    confidence_scores = torch.softmax(predictions, dim=1).cpu().numpy()[0] * 100
    sorted_indices = np.argsort(confidence_scores)[::-1]
    class_labels = ['Melanoma', 'Nevus', 'Benign Keratosis', 'Basal Cell Carcinoma', 'Actinic Keratosis', 'Vascular lesions', 'Dermatofibroma', 'Squamous Cell Carcinoma']

    print("\nPrediction Confidence Scores:")
    for idx in sorted_indices:
        print(f"{class_labels[idx]}: {confidence_scores[idx]:.2f}%")

    predicted_class = class_labels[sorted_indices[0]]
    
    predicted_confidence = confidence_scores[sorted_indices[0]]
    # Get description of the predicted class
    lesion_info = skin_lesion_info[predicted_class]

    # Legal Disclaimer
    disclaimer = "This is an AI-based skin lesion classifier. This result does NOT constitute a medical diagnosis. Always consult a **qualified dermatologist or healthcare provider** for an accurate assessment."

    return predicted_class, predicted_confidence,lesion_info, disclaimer

# Function to generate a PDF report with user inputs

def generate_pdf(predicted_class, predicted_confidence, lesion_info, age, gender, region, image_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, "Skin Lesion Classification Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Age: {age}", ln=True)
    pdf.cell(0, 10, f"Gender: {gender}", ln=True)
    pdf.cell(0, 10, f"Region: {region}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, f"Predicted Condition: {predicted_class}", ln=True)
    pdf.cell(0, 10, f"Confidence: {predicted_confidence*100:.2f}%", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, f"Condition Info:\n{lesion_info}")
    pdf.ln(5)

    pdf.set_font("Arial", style="I", size=10)
    pdf.set_text_color(255, 0, 0)
    pdf.multi_cell(0, 8, f"Disclaimer: This is an AI-based skin lesion classifier. This result does NOT constitute a medical diagnosis. Always consult a **qualified dermatologist or healthcare provider** for an accurate assessment")

    if os.path.exists(image_path):
        pdf.image(image_path, x=60, w=90)

    pdf.set_text_color(0, 0, 0)
    file_path = "skin_lesion_report.pdf"
    pdf.output(file_path)
    return file_path

# Gradio interface function
def gradio_interface(image, age, gender, region):
    image_path = "temp_uploaded_image.jpg"
    image.save(image_path)
    
    predicted_class, predicted_confidence, lesion_info, disclaimer = classify_image(image_path, age, gender, region)
    pdf_path = generate_pdf(predicted_class, predicted_confidence, lesion_info, age, gender, region, image_path)

    # Output formatting
    confidence_color = "green" if predicted_confidence >= 60 else "red"
    confidence_text = ("High confidence in prediction." if predicted_confidence >= 60 
                       else "Low confidence in prediction. Interpret with caution.")
    confidence_box = (
        f"### Confidence: {predicted_confidence:.2f}%\n"
        f"<span style='color:{confidence_color};font-weight:bold;'>{confidence_text}</span>"
    )
    condition_box = f"### Condition: {predicted_class}"
    info_box = f"### Definition\n{lesion_info}"
    disclaimer_box = f"### Disclaimer\n{disclaimer}"

    return gr.Markdown(condition_box), gr.Markdown(confidence_box), gr.Markdown(info_box), gr.Markdown(disclaimer_box),pdf_path

# Sample Images to show in Gallery
sample_images = [
    ("/Users/hannah/Desktop/BDBA/FOURTH YEAR/Capstone Project/IMAGES/ISIC-images-2019-test/ISIC_0035393.jpg", "Melanoma"),
    ("/Users/hannah/Desktop/BDBA/FOURTH YEAR/Capstone Project/IMAGES/ISIC-images-2019-test/ISIC_0035396.jpg", "Nevus"),
    ("/Users/hannah/Desktop/BDBA/FOURTH YEAR/Capstone Project/IMAGES/ISIC-images-2019-test/ISIC_0035733.jpg", "Basal Cell Carcinoma"),
]  


# Gradio UI
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("""# Skin Lesion Classifier
    Upload an image or use a sample image to test the tool. Enter **age, gender, and lesion region** for an AI-based classification. This tool uses ViT and EfficientNet pretrained models in a CNN architecture trained on dermatological images.
    """)
    with gr.Row():
        with gr.Column():
            # Dropdown for sample images
            sample_image_dropdown = gr.Dropdown(
                choices=[img[1] for img in sample_images], 
                label="Select a Sample Image"
            )
            image_input = gr.Image(type="pil", label="Upload or Select Lesion Image")
            age_input = gr.Number(label="Age", value=55)
            gender_input = gr.Radio(label="Gender", choices=["Male", "Female", "Other"])
            region_input = gr.Textbox(label="Region (e.g. abdomen, back, arm)")
            submit_btn = gr.Button("Classify")

        with gr.Column():
            condition_output = gr.Markdown(label="Predicted Condition")
            confidence_output = gr.Markdown(label='Confidence Score')
            info_output = gr.Markdown(label="Condition Info")
            pdf_output = gr.File(label="Download Report")
            disclaimer_output = gr.Markdown(label="Disclaimer")

    # Update image input when a sample image is selected
    def update_image(selected_sample):
        for img_path, label in sample_images:
            if label == selected_sample:
                return Image.open(img_path)

    sample_image_dropdown.change(
        fn=update_image,
        inputs=[sample_image_dropdown],
        outputs=[image_input]
    )

    submit_btn.click(
        fn=gradio_interface,
        inputs=[image_input, age_input, gender_input, region_input],
        outputs=[condition_output, confidence_output, info_output, disclaimer_output, pdf_output]
    )

# Launch
demo.launch()