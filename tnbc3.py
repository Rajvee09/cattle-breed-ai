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

        self.weight = nn.Parameter(
            torch.ones(channels)
        )

        self.bias = nn.Parameter(
            torch.zeros(channels)
        )

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

        # Stage 0
        self.stages.append(
            Stage(96, 3)
        )

        # Stage 1
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

            "blocks": Stage(
                192,
                3
            ).blocks

        })

        self.stages.append(stage1)

        # Stage 2
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

            "blocks": Stage(
                384,
                9
            ).blocks

        })

        self.stages.append(stage2)

        # Stage 3
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

            "blocks": Stage(
                768,
                3
            ).blocks

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

        x = self.stem(x)

        # Stage 0
        x = self.stages[0](x)

        # Stage 1
        x = self.stages[1]["downsample"](x)

        for block in self.stages[1]["blocks"]:

            x = block(x)

        # Stage 2
        x = self.stages[2]["downsample"](x)

        for block in self.stages[2]["blocks"]:

            x = block(x)

        # Stage 3
        x = self.stages[3]["downsample"](x)

        for block in self.stages[3]["blocks"]:

            x = block(x)

        # Global average pooling
        x = x.mean(dim=(-2, -1))

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


print("\n========================================")
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
# 12. BREED PROFILE DATABASE
# ============================================================

BREED_PROFILES = {

    "Alambadi": (
        "Tamil Nadu, India",
        "Draught",
        "Farm work and transport",
        "Grey/dark-grey coat; compact and strong build"
    ),

    "Amritmahal": (
        "Karnataka, India",
        "Draught",
        "Farm work and transport",
        "Grey coat; long head and distinctive long horns"
    ),

    "Ayrshire": (
        "Scotland",
        "Dairy",
        "Milk production",
        "Red-and-white coat; medium-sized dairy cattle"
    ),

    "Banni": (
        "Kutch, Gujarat, India",
        "Buffalo",
        "Milk production",
        "Mostly black coat; large upward-curving horns"
    ),

    "Bargur": (
        "Tamil Nadu, India",
        "Draught",
        "Farm work",
        "Compact and sturdy; adapted to hilly terrain"
    ),

    "Bhadawari": (
        "Uttar Pradesh / Madhya Pradesh, India",
        "Buffalo",
        "Milk production",
        "Copper/brown coat; hardy buffalo"
    ),

    "Brown_Swiss": (
        "Switzerland",
        "Dairy",
        "Milk production",
        "Brown-grey coat; strong dairy build"
    ),

    "Dangi": (
        "Maharashtra / Madhya Pradesh, India",
        "Draught",
        "Farm work",
        "Hardy indigenous cattle suited to difficult terrain"
    ),

    "Deoni": (
        "Maharashtra / Karnataka, India",
        "Dual purpose",
        "Milk and farm work",
        "Black-and-white spotted coat"
    ),

    "Gir": (
        "Gujarat, India",
        "Dairy",
        "Milk production",
        "Red/red-spotted coat; prominent forehead; long ears; curved horns"
    ),

    "Guernsey": (
        "Guernsey",
        "Dairy",
        "Milk production",
        "Fawn/red-and-white coat"
    ),

    "Hallikar": (
        "Karnataka, India",
        "Draught",
        "Farm work",
        "Grey coat; strong body and upward-curving horns"
    ),

    "Hariana": (
        "Haryana / Uttar Pradesh / Rajasthan, India",
        "Dual purpose",
        "Milk and farm work",
        "White/light-grey coat"
    ),

    "Holstein_Friesian": (
        "Netherlands / Northern Europe",
        "Dairy",
        "Milk production",
        "Black-and-white coat; high-yield dairy breed"
    ),

    "Jaffrabadi": (
        "Gujarat, India",
        "Buffalo",
        "Milk production",
        "Large heavy buffalo with strongly curved horns"
    ),

    "Jersey": (
        "Jersey",
        "Dairy",
        "Milk production",
        "Fawn-brown coat; compact dairy build"
    ),

    "Kangayam": (
        "Tamil Nadu, India",
        "Draught",
        "Farm work and transport",
        "Grey/white coat; strong and sturdy build"
    ),

    "Kankrej": (
        "Gujarat / Rajasthan, India",
        "Dual purpose",
        "Milk and farm work",
        "Silver/steel-grey coat; lyre-shaped horns"
    ),

    "Kasargod": (
        "Kerala, India",
        "Dairy",
        "Milk production",
        "Small indigenous cattle; hardy and low-input"
    ),

    "Kenkatha": (
        "Uttar Pradesh / Madhya Pradesh, India",
        "Draught",
        "Farm work",
        "Small and sturdy indigenous cattle"
    ),

    "Kherigarh": (
        "Uttar Pradesh, India",
        "Draught",
        "Farm work",
        "White coat with brown markings; upward/outward horns"
    ),

    "Khillari": (
        "Maharashtra / Karnataka, India",
        "Draught",
        "Farm work and transport",
        "Compact and powerful build"
    ),

    "Krishna_Valley": (
        "Karnataka / Maharashtra, India",
        "Dual purpose",
        "Milk and farm work",
        "Large grey-white cattle"
    ),

    "Malnad_gidda": (
        "Karnataka, India",
        "Dual purpose",
        "Milk and local utility",
        "Small cattle adapted to hilly, high-rainfall areas"
    ),

    "Mehsana": (
        "Gujarat, India",
        "Buffalo",
        "Milk production",
        "Black coat; medium-large body; curved horns"
    ),

    "Murrah": (
        "Haryana / North-West India",
        "Buffalo",
        "Milk production",
        "Black compact body; tightly curled horns"
    ),

    "Nagori": (
        "Rajasthan, India",
        "Draught",
        "Farm work and transport",
        "Light-bodied, strong draught cattle"
    ),

    "Nagpuri": (
        "Maharashtra, India",
        "Buffalo",
        "Milk and farm utility",
        "Black coat; adapted to hot and dry conditions"
    ),

    "Nili_Ravi": (
        "Punjab region",
        "Buffalo",
        "Milk production",
        "Black coat with white markings; curled horns"
    ),

    "Nimari": (
        "Madhya Pradesh, India",
        "Dual purpose",
        "Milk and farm work",
        "Hardy indigenous dual-purpose cattle"
    ),

    "Ongole": (
        "Andhra Pradesh, India",
        "Draught",
        "Farm work and transport",
        "Large white/grey cattle with prominent hump"
    ),

    "Pulikulam": (
        "Tamil Nadu, India",
        "Draught",
        "Farm work and traditional systems",
        "Hardy indigenous cattle"
    ),

    "Rathi": (
        "Rajasthan, India",
        "Dual purpose",
        "Milk and farm work",
        "Brown-and-white spotted coat"
    ),

    "Red_Dane": (
        "Denmark",
        "Dairy",
        "Milk production",
        "Red coat; dairy-type cattle"
    ),

    "Red_Sindhi": (
        "Sindh region",
        "Dairy",
        "Milk production",
        "Deep red coat; hardy dairy cattle"
    ),

    "Sahiwal": (
        "Punjab region / Rajasthan",
        "Dairy",
        "Milk production",
        "Reddish-brown coat; loose skin; heat-adapted"
    ),

    "Surti": (
        "Gujarat, India",
        "Buffalo",
        "Milk production",
        "Black/grey coat; crescent-shaped horns"
    ),

    "Tharparkar": (
        "Rajasthan, India",
        "Dual purpose",
        "Milk and farm work",
        "White/light-grey coat; adapted to hot, dry conditions"
    ),

    "Toda": (
        "Nilgiri Hills, Tamil Nadu, India",
        "Dairy",
        "Milk production",
        "Distinctive hill cattle"
    ),

    "Umblachery": (
        "Tamil Nadu, India",
        "Draught",
        "Farm work",
        "Small-medium cattle suited to wet areas"
    ),

    "Vechur": (
        "Kerala, India",
        "Dairy",
        "Milk and low-input farming",
        "Very small indigenous cattle"
    )
}


# ============================================================
# 13. GET BREED PROFILE
# ============================================================

def get_breed_profile(breed_name):

    profile = BREED_PROFILES.get(breed_name)

    if profile is None:

        return {
            "origin": "Information not available",
            "type": "Information not available",
            "use": "Information not available",
            "features": "Information not available"
        }

    return {

        "origin": profile[0],

        "type": profile[1],

        "use": profile[2],

        "features": profile[3]

    }


# ============================================================
# 14. PREDICT BREED
# ============================================================

def predict_breed(image):

    print("\nAnalyzing image...")

    # --------------------------------------------------------
    # SUPPORT PIL IMAGE / FILE PATH / FILE OBJECT
    # --------------------------------------------------------

    if not isinstance(image, Image.Image):

        image = Image.open(image)

    # Convert JPG / PNG / WEBP etc. to RGB
    image = image.convert("RGB")

    # --------------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------------

    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move tensor to CPU/GPU
    image_tensor = image_tensor.to(device)

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # BUILD RESULTS
    # --------------------------------------------------------

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

        # Get breed information
        profile = get_breed_profile(
            breed
        )

        # Add prediction + breed information
        results.append({

            "breed": breed,

            "confidence": round(
                confidence,
                2
            ),

            "origin": profile["origin"],

            "type": profile["type"],

            "use": profile["use"],

            "features": profile["features"]

        })

    return results


# ============================================================
# 15. TEST FROM TERMINAL
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

            print(
                f"   Origin: "
                f"{result['origin']}"
            )

            print(
                f"   Type: "
                f"{result['type']}"
            )

            print(
                f"   Main Use: "
                f"{result['use']}"
            )

            print(
                f"   Key Features: "
                f"{result['features']}"
            )

        print("\n========================================")
        print("       PREDICTION COMPLETE")
        print("========================================")