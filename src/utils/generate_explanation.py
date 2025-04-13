import torch
from torchvision import transforms
from torchvision.models import resnet18
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image
import matplotlib.pyplot as plt
import os

# Preprocesamiento idéntico al usado en el entrenamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def generate_cam(model_path, image_path, output_path="reports/figures/cam_result.pg"):
    # Verificar rutas
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Debug: Mostrar rutas
    print(f"Model path: {model_path}\nImage path: {image_path}\nOutput path: {output_path}")

    # Cargar modelo (con verificación)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    try:
        model = torch.load(model_path, map_location='cpu')
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error cargando modelo: {e}")

    # Resto del código...
    try:
        img = Image.open(image_path).convert("RGB")
        print(f"Tamaño imagen original: {img.size}")  # Debug
        input_tensor = transform(img).unsqueeze(0)
        
        # Debug: Verificar tensor
        print(f"Rango tensor: {input_tensor.min().item():.3f} - {input_tensor.max().item():.3f}")
        
        output = model(input_tensor)
        class_id = output.argmax().item()
        print(f"Clase predicha: {class_id}")  # Debug
        
        # Generar y guardar CAM
        cam_extractor = GradCAM(model, target_layer='layer4')
        activation_map = cam_extractor(class_id, output)[0]
        result = overlay_mask(img, activation_map, alpha=0.5)
        
        result.save(output_path)
        print(f"✅ Heatmap guardado en: {output_path}")
        
        return output_path, class_id
    except Exception as e:
        raise RuntimeError(f"Error generando CAM: {e}")
