import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePointNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # Shared MLP over points
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x: (B, N, 3)
        returns: logits (B, num_classes)
        """
        # Apply shared MLP to each point
        # nn.Linear works on last dimension, so this is fine
        x = self.mlp1(x)              # (B, N, 256)

        # Symmetric aggregation: max-pool over points
        x = torch.max(x, dim=1)[0]    # (B, 256)

        # Classifier
        x = self.classifier(x)        # (B, num_classes)
        return x


def test_model():
    print("=" * 70)
    print("Step 12: Testing SimplePointNet")
    print("=" * 70)

    model = SimplePointNet(num_classes=3)
    print(model)

    batch_size = 8
    num_points = 1024
    num_dims = 3

    x = torch.randn(batch_size, num_points, num_dims)
    logits = model(x)

    print("\nInput shape:")
    print(x.shape)

    print("\nOutput logits shape:")
    print(logits.shape)

    print("\nSample output logits:")
    print(logits[:2])

    assert logits.shape == (batch_size, 3), "Output shape is incorrect."

    print("\nStep 12 passed: model forward pass works correctly.")


if __name__ == "__main__":
    test_model()
