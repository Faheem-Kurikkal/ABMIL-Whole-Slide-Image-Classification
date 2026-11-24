
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class RegularizedABMIL(nn.Module):

    def __init__(
        self,
        input_dim=512,
        hidden_dim=128,
        num_classes=2,
        dropout=0.35
    ):
        super().__init__()

        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )

        self.attention_weights = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, features):

        # Gated attention
        A_V = self.attention_V(features)
        A_U = self.attention_U(features)

        A = A_V * A_U

        attention_scores = self.attention_weights(A)

        # Normalize attention across patches
        attention = torch.softmax(
            attention_scores.squeeze(-1),
            dim=0
        )

        # Weighted aggregation
        bag_features = torch.sum(
            attention.unsqueeze(-1) * features,
            dim=0
        )

        # Slide-level classification
        logits = self.classifier(bag_features)

        return logits, attention


def load_models(model_dir, device):

    # Load ResNet18 feature extractor
    encoder = resnet18(weights=None)
    encoder.fc = nn.Identity()

    encoder_path = f"{model_dir}/resnet18_encoder.pth"

    encoder.load_state_dict(
        torch.load(
            encoder_path,
            map_location=device,
            weights_only=True
        )
    )

    encoder = encoder.to(device)
    encoder.eval()

    # Load trained ABMIL classifier
    mil_model = RegularizedABMIL(
        input_dim=512,
        hidden_dim=128,
        num_classes=2,
        dropout=0.35
    )

    checkpoint_path = f"{model_dir}/abmil_regularized_best.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    mil_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    mil_model = mil_model.to(device)
    mil_model.eval()

    # Exact ResNet18 ImageNet preprocessing
    transform = ResNet18_Weights.DEFAULT.transforms()

    return encoder, mil_model, transform
