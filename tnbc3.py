import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# 2. LOAD FILES FROM HUGGING FACE
# ============================================================

REPO_ID = "ujjwal75/indian-bovine-breeds-model"

print("\nLoading model files...")

classes_file_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="classes.json"
)

model_weight_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="Indian_bovine_finetuned_model.pth"
)

print("Classes file:", classes_file_path)
print("Model file:", model_weight_path)


# ============================================================
# 3. LOAD BREED NAMES
# ============================================================

with open(classes_file_path, "r") as f:
    CLASS_NAMES = json.load(f)

print("\nNumber of breeds:", len(CLASS_NAMES))


# ============================================================
# 4. LAYER NORM FOR IMAGE TENSORS
# ============================================================

class LayerNorm2d(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.channels = channels

    def forward(self, x):

        # N,C,H,W -> N,H,W,C
        x = x.permute(0, 2, 3, 1)

        x = nn.functional.layer_norm(
            x,
            (self.channels,),
            self.weight,
            self.bias
        )

        # N,H,W,C -> N,C,H,W
        x = x.permute(0, 3, 1, 2)

        return x


# ============================================================
# 5. CONVNEXT BLOCK
# ============================================================

class ConvNeXtBlock(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.gamma = nn.Parameter(
            torch.ones(dim) * 1e-6
        )

        self.conv_dw = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim
        )

        self.norm = nn.LayerNorm(dim)

        self.mlp = nn.ModuleDict({
            "fc1": nn.Linear(
                dim,
                4 * dim
            ),

            "fc2": nn.Linear(
                4 * dim,
                dim
            )
        })

        self.act = nn.GELU()

    def forward(self, x):

        shortcut = x

        x = self.conv_dw(x)

        # N,C,H,W -> N,H,W,C
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)

        x = self.mlp["fc1"](x)

        x = self.act(x)

        x = self.mlp["fc2"](x)

        x = x * self.gamma

        # N,H,W,C -> N,C,H,W
        x = x.permute(0, 3, 1, 2)

        return shortcut + x


# ============================================================
# 6. STAGE
# ============================================================

class Stage(nn.Module):

    def __init__(self, dim, depth):

        super().__init__()

        self.blocks = nn.ModuleList([
            ConvNeXtBlock(dim)
            for _ in range(depth)
        ])

    def forward(self, x):

        for block in self.blocks:
            x = block(x)

        return x


# ============================================================
# 7. COMPLETE CONVNEXT-TINY
# ============================================================

class ConvNeXtTiny(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        # ----------------------------------------------------
        # STEM
        # ----------------------------------------------------

        self.stem = nn.Sequential(

            nn.Conv2d(
                3,
                96,
                kernel_size=4,
                stride=4
            ),

            LayerNorm2d(96)
        )


        # ----------------------------------------------------
        # STAGES
        # ----------------------------------------------------

        self.stages = nn.ModuleList()


        # ----------------------------------------------------
        # Stage 0
        # ----------------------------------------------------

        self.stages.append(
            Stage(96, 3)
        )


        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------

        stage1 = nn.ModuleDict({

            "downsample": nn.Sequential(

                LayerNorm2d(96),

                nn.Conv2d(
                    96,
                    192,
                    kernel_size=2,
                    stride=2
                )
            ),

            # IMPORTANT:
            # Direct ModuleList, NOT Stage()
            # This matches checkpoint keys:
            # stages.1.blocks.0
            "blocks": nn.ModuleList([
                ConvNeXtBlock(192)
                for _ in range(3)
            ])
        })

        self.stages.append(stage1)


        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------

        stage2 = nn.ModuleDict({

            "downsample": nn.Sequential(

                LayerNorm2d(192),

                nn.Conv2d(
                    192,
                    384,
                    kernel_size=2,
                    stride=2
                )
            ),

            # Direct ModuleList
            "blocks": nn.ModuleList([
                ConvNeXtBlock(384)
                for _ in range(9)
            ])
        })

        self.stages.append(stage2)


        # ----------------------------------------------------
        # Stage 3
        # ----------------------------------------------------

        stage3 = nn.ModuleDict({

            "downsample": nn.Sequential(

                LayerNorm2d(384),

                nn.Conv2d(
                    384,
                    768,
                    kernel_size=2,
                    stride=2
                )
            ),

            # Direct ModuleList
            "blocks": nn.ModuleList([
                ConvNeXtBlock(768)
                for _ in range(3)
            ])
        })

        self.stages.append(stage3)


        # ----------------------------------------------------
        # HEAD
        # ----------------------------------------------------

        self.head = nn.ModuleDict({

            "norm": nn.LayerNorm(768),

            "fc": nn.Linear(
                768,
                num_classes
            )
        })


    def forward(self, x):

        # Stem
        x = self.stem(x)


        # ----------------------------------------------------
        # Stage 0
        # ----------------------------------------------------

        x = self.stages[0](x)


        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------

        x = self.stages[1]["downsample"](x)

        for block in self.stages[1]["blocks"]:
            x = block(x)


        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------

        x = self.stages[2]["downsample"](x)

        for block in self.stages[2]["blocks"]:
            x = block(x)


        # ----------------------------------------------------
        # Stage 3
        # ----------------------------------------------------

        x = self.stages[3]["downsample"](x)

        for block in self.stages[3]["blocks"]:
            x = block(x)


        # ----------------------------------------------------
        # Global average pooling
        # ----------------------------------------------------

        x = x.mean(
            dim=(-2, -1)
        )


        # ----------------------------------------------------
        # HEAD
        # ----------------------------------------------------

        x = self.head["norm"](x)

        x = self.head["fc"](x)

        return x


# ============================================================
# 8. CREATE MODEL
# ============================================================

print("\nCreating ConvNeXt-Tiny...")

model = ConvNeXtTiny(
    num_classes=len(CLASS_NAMES)
)


# ============================================================
# 9. LOAD CHECKPOINT
# ============================================================

print("Loading trained weights...")

checkpoint = torch.load(
    model_weight_path,
    map_location="cpu"
)

state_dict = checkpoint["model_state_dict"]


# Remove "module." if present

cleaned_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[7:]

    cleaned_state_dict[key] = value


# ============================================================
# 10. LOAD WEIGHTS
# ============================================================

model.load_state_dict(
    cleaned_state_dict,
    strict=True
)

model = model.to(device)

model.eval()


print("\n")
print("========================================")
print("       MODEL LOADED SUCCESSFULLY!")
print("========================================")


# ============================================================
# 11. IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# 12. PREDICT BREED
# ============================================================

def predict_breed(image_path):

    print("\nAnalyzing image...")

    image = Image.open(
        image_path
    ).convert("RGB")


    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)


    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        top_probs, top_indices = torch.topk(
            probabilities,
            k=3,
            dim=1
        )


    results = []


    for probability, index in zip(
        top_probs[0],
        top_indices[0]
    ):

        breed = CLASS_NAMES[
            index.item()
        ]

        confidence = (
            probability.item() * 100
        )


        results.append({

            "breed": breed,

            "confidence": confidence

        })


    return results


# ============================================================
# 13. RUN PREDICTION
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("========================================")
    print("   INDIAN BOVINE BREED IDENTIFIER")
    print("========================================")


    image_path = input(
        "\nEnter image filename/path: "
    ).strip()


    if not os.path.exists(image_path):

        print("\nERROR: Image not found!")

        print(
            "Check the filename/path and try again."
        )

    else:

        predictions = predict_breed(
            image_path
        )


        print("\n")
        print("========================================")
        print("          TOP 3 PREDICTIONS")
        print("========================================")


        for i, result in enumerate(
            predictions,
            start=1
        ):

            print(
                f"\n{i}. {result['breed']}"
            )

            print(
                f"   Confidence: "
                f"{result['confidence']:.2f}%"
            )


        print("\n")
        print("========================================")
        print("       PREDICTION COMPLETE")
        print("========================================")
