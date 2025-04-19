import torchvision
import torchvision.transforms as transforms # Needed even if not used directly here

# Define a dummy transform just to satisfy the API,
# actual transforms happen during training.
transform = transforms.ToTensor()

# Download unlabeled data (for pretraining)
unlabeled_dataset = torchvision.datasets.STL10(
    root='./data',
    split='unlabeled',
    download=True,
    transform=transform # A transform is required, even if basic
)
print(f"Downloaded {len(unlabeled_dataset)} unlabeled images.")

# Download labeled training data (for linear probing / k-NN train set)
train_dataset = torchvision.datasets.STL10(
    root='./data',
    split='train',
    download=True,
    transform=transform
)
print(f"Downloaded {len(train_dataset)} labeled train images.")

# Download labeled test data (for linear probing / k-NN test set)
test_dataset = torchvision.datasets.STL10(
    root='./data',
    split='test',
    download=True,
    transform=transform
)
print(f"Downloaded {len(test_dataset)} labeled test images.")

print("STL-10 dataset downloaded to ./data")