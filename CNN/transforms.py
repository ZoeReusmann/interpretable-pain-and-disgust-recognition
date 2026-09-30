from torchvision import transforms
from constants import IMAGE_SIZE   


def get_default_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # FaceNet erwartet 160x160
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

train_transform = get_default_transform()