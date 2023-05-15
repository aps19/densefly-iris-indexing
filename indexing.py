import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from collections import defaultdict
from PIL import Image

# DenseFly and PseudoHash functions
def rand(a, b):
    """Generate a set of a random integers in [0, b]"""
    return np.random.choice(b, a, replace=False)

def generate_projections(d, m, k, alpha):
    """Generate mk sparse, binary random projections"""
    S = [rand(int(alpha*d), d) for _ in range(m*k)]
    return S

def dense_fly(v, m, k, alpha):
    d = len(v)
    S = generate_projections(d, m, k, alpha)

    activations = np.zeros(m*k)
    for j in range(m*k):
        activations[j] = np.sum(v[S[j]])

    h1 = np.sign(activations)
    h1 = (h1 + 1) / 2  # convert to binary {0, 1}
    return h1

def pseudo_hash(h1, m, k):
    h2 = np.zeros(m)
    for j in range(m):
        h2[j] = np.sign(np.mean(h1[k*j:k*(j+1)]) - 0.5)
    h2 = (h2 + 1) / 2  # convert to binary {0, 1}
    return h2

# Load the pre-trained ResNet18 model
model = resnet18(pretrained=True)
model.eval()

# Set up image preprocessing
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Load json file containing the extracted feature vectors
with open('iris_data.json', 'r') as f:
    data = json.load(f)

# Initialize parameters for DenseFly and PseudoHash
m = 16
k = 4
alpha = 0.1

# Generate hash codes for all images
hash_table = defaultdict(list)
for img_id, img_path in data.items():
    img = Image.open(img_path)
    input_tensor = preprocess(img).unsqueeze(0)
    feature_vector = model(input_tensor).detach().numpy()

    h1 = dense_fly(feature_vector, m, k, alpha)
    h2 = pseudo_hash(h1, m, k)

    hash_key = tuple(h2)
    hash_table[hash_key].append((img_id, h1))

def find_nearest_neighbors(query_img_path, hash_table, m, k, alpha, top_n=5, tau=5):
    query_img = Image.open(query_img_path)
    input_tensor = preprocess(query_img).unsqueeze(0)
    query_feature_vector = model(input_tensor).detach().numpy()

    query_h1 = dense_fly(query_feature_vector, m, k, alpha)
    query_h2 = pseudo_hash(query_h1, m, k)

    candidates = []
    for img_id, h1 in hash_table[tuple(query_h2)]:
        hamming_distance = np.sum(np.abs(query_h1 - h1))
        if hamming_distance <= tau:
            candidates.append((img_id, hamming_distance))

    candidates.sort(key=lambda x: x[1])

    # Get the top n candidates
    top_n_images = candidates[:top_n]

    return top_n_images


# Test the function with a sample query image
query_img_path = 'path_to_your_query_image.jpg'  # replace with your query image path
top_n_neighbors = find_nearest_neighbors(query_img_path, hash_table, m, k, alpha, top_n=5, tau=5)

# Print out the results
for img_id, distance in top_n_neighbors:
    print(f"Image ID: {img_id}, Hamming Distance: {distance}")
