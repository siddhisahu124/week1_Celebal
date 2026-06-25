import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

CLASSES = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

FEATURE_EXTRACTOR = None


def load_model(model_path):

    global FEATURE_EXTRACTOR

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASSES)
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu"
        )
    )

    model.eval()

    FEATURE_EXTRACTOR = nn.Sequential(
        *list(model.children())[:-1]
    )

    FEATURE_EXTRACTOR.eval()

    return model


def predict_image(model, image):

    image = image.convert("RGB")

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():

        output = model(tensor)

        probs = torch.softmax(
            output,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            1
        )

    return (
        CLASSES[pred.item()],
        confidence.item()
    )


def get_embedding(image):

    image = image.convert("RGB")

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():

        embedding = FEATURE_EXTRACTOR(
            tensor
        )

    embedding = embedding.squeeze()

    return embedding.numpy()


def compute_similarity(
    image1,
    image2
):

    emb1 = get_embedding(image1)

    emb2 = get_embedding(image2)

    similarity = cosine_similarity(
        [emb1],
        [emb2]
    )[0][0]

    return similarity


def detect_change(
    image1,
    image2,
    threshold
):

    similarity = compute_similarity(
        image1,
        image2
    )

    changed = (
        similarity < threshold
    )

    return similarity, changed


def generate_heatmap(
    image1,
    image2
):

    img1 = np.array(
        image1.resize((224, 224))
    ).astype(np.float32)

    img2 = np.array(
        image2.resize((224, 224))
    ).astype(np.float32)

    diff = np.abs(
        img1 - img2
    )

    heatmap = diff.mean(
        axis=2
    )

    return heatmap