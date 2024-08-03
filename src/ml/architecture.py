from datasets import load_dataset
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2


class ConvolutionalNN(nn.Module):
    def __init__(self):
        super().__init__()

        dim = 2 ** 6  # w, h of images
        p_dropout = 0.05
        out_channels = 32
        fc_features = out_channels * dim * dim // ((4 * 4) * (2 * 2) * (2 * 2))

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, stride=2, padding=2),  # -> 48 x dim/2 x dim/2
            nn.ReLU(),
            nn.BatchNorm2d(num_features=48),
            nn.MaxPool2d(kernel_size=2, stride=2),  # dim/4 x dim/4

            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding="same"),
            # -> 96 x dim/4 x dim/4
            nn.ReLU(),
            nn.BatchNorm2d(num_features=96),
            nn.MaxPool2d(kernel_size=2, stride=2),  # dim/8 x dim/8

            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding="same"),
            # -> 128 x dim/8 x dim/8
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding="same"),
            # -> 128 x dim/8 x dim/8
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),

            nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            # -> out_channels x dim/8 x dim/8
            nn.ReLU(),
            # No BatchNorm before first fc layer
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> out_channels x dim/16 x dim/16

            nn.Flatten(),  # -> 8192

            nn.Linear(in_features=fc_features, out_features=fc_features // 4),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.BatchNorm1d(fc_features // 4),

            nn.Linear(in_features=fc_features // 4, out_features=fc_features // 16),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.BatchNorm1d(fc_features // 16),

            nn.Linear(in_features=fc_features // 16, out_features=3),
        )
        self._initialize_weights()

    def forward(self, x):
        return self.cnn(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# https://huggingface.co/transformers/v3.2.0/custom_datasets.html
# https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
class ImageDataset(torch.utils.data.Dataset):

    src_dataset = None  # Singleton of train + valid + test

    def __init__(self, dim, split="train", is_grayscale=False, augmented=False, policies=None, images_per_policy=0):
        """
        :param dim: w, h s.t. w = h
        :param split: train | valid | test
        :param is_grayscale:
        :param augmented:
        :param policies:
        :param images_per_policy:
        """
        if ImageDataset.src_dataset is None:
            ImageDataset.src_dataset = load_dataset("camfruss/bread_proofing")

        self.df = pd.DataFrame(ImageDataset.src_dataset[split])
        self.dim = dim
        self.is_grayscale = is_grayscale
        self.augmented = augmented
        self.policies = policies
        self.images_per_policy = images_per_policy

        if augmented and policies:
            self.augmentation_factor = len(policies) * images_per_policy
        else:
            self.augmentation_factor = 0

    def __getitem__(self, idx):
        true_idx = idx // (self.augmentation_factor + 1) if self.augmented else idx

        image = self.df.iloc[true_idx]["image"]
        if self.augmented:
            image = self.augment_image(image, idx)
        else:
            image = self.get_transforms()(image)

        label = self.df["label"].iloc[true_idx]
        return image, label

    def __len__(self):
        if self.augmented:
            return len(self.df) * (self.augmentation_factor + 1)
        return len(self.df)

    def augment_image(self, image, idx):
        tid = idx % (self.augmentation_factor + 1)  # transformation ID
        if tid == 0:
            return self.get_transforms()(image)

        policy_idx = (tid - 1) // self.images_per_policy
        if policy_idx >= len(self.policies):
            raise Exception("Unexpected data augmentation error")

        tail = v2.AutoAugment(policy=self.policies[policy_idx])
        image = self.get_transforms(tail=tail)(image)
        return image

    def get_transforms(self, tail=None):
        """
        :param tail: list of additional transforms to add
        """
        transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((self.dim, self.dim))
        ]

        if self.is_grayscale:
            transforms.append(v2.Grayscale())
        if tail is not None:
            transforms.append(tail)

        transforms = v2.Compose(transforms)
        return transforms


# def create_dl(self, df, augmented=False):
#     return DataLoader(BreadProofingDataset(df, augmented=augmented), batch_size=batch_size, shuffle=True)

# https://huggingface.co/transformers/v3.2.0/custom_datasets.html
# https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py

# policies = [
#     v2.AutoAugmentPolicy.CIFAR10,
#     v2.AutoAugmentPolicy.IMAGENET,
#     v2.AutoAugmentPolicy.SVHN
# ]
#
# images_per_policy = 4
# augmentation_factor = len(policies) * images_per_policy  # currently 12
#
# transforms = v2.Compose([
#     v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
#     # v2.Grayscale(),
#     v2.Resize((dim, dim)),
#     # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# transforms = v2.Compose([
#     v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
#     # v2.Grayscale(),
#     v2.Resize((dim, dim)),
# ])


#
# class BreadProofingDataset(torch.utils.data.Dataset):
#
#     def __init__(self, df, augmented=False):
#         self.df = df
#         self.augmented = augmented
#
#     def __getitem__(self, idx):
#         true_idx = idx // (augmentation_factor + 1) if self.augmented else idx
#
#         image = self.df.iloc[true_idx]["image"]
#         if self.augmented:
#             image = augment_image(image, idx)
#         else:
#             image = transforms(image)
#
#         label = self.df["label"].iloc[true_idx]
#         # label = torch.tensor(label, dtype=torch.long)  # CrossEntropyLoss expects class indices
#         # label = F.one_hot(label, num_classes=3).float()
#
#         return image, label
#
#     def __len__(self):
#         if self.augmented:
#             return len(self.df) * (augmentation_factor + 1)
#         return len(self.df)


# label = torch.tensor(label, dtype=torch.long)  # CrossEntropyLoss expects class indices
# label = F.one_hot(label, num_classes=3).float()

# train_df = pd.DataFrame(dataset["train"])
# valid_df = pd.DataFrame(dataset["valid"])
# test_df = pd.DataFrame(dataset["test"])

# policies = [
#     v2.AutoAugmentPolicy.CIFAR10,
#     v2.AutoAugmentPolicy.IMAGENET,
#     v2.AutoAugmentPolicy.SVHN
# ]
#
# images_per_policy = 4
# augmentation_factor = len(policies) * images_per_policy  # currently 12
#
# transforms = v2.Compose([
#     v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
#     # v2.Grayscale(),
#     v2.Resize((dim, dim)),
#     # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
#
# def augment_image(image, idx):
#     tid = idx % (augmentation_factor + 1)  # transformation ID
#
#     if tid == 0:
#         return transforms(image)
#
#     policy_idx = (tid - 1) // images_per_policy
#     if policy_idx >= len(policies):
#         raise Exception("Unexpected data augmentation error")
#
#     train_transforms = v2.Compose([
#         v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
#         # v2.Grayscale(),
#         v2.Resize((dim, dim)),
#         v2.AutoAugment(policy=policies[policy_idx]),  # augmentation before normalization
#         # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     return train_transforms(image)
#
#
# class BreadProofingDataset(torch.utils.data.Dataset):
#
#     def __init__(self, df, augmented=False):
#         self.df = df
#         self.augmented = augmented
#
#     def __getitem__(self, idx):
#         true_idx = idx // (augmentation_factor + 1) if self.augmented else idx
#
#         image = self.df.iloc[true_idx]["image"]
#         if self.augmented:
#             image = augment_image(image, idx)
#         else:
#             image = transforms(image)
#
#         label = self.df["label"].iloc[true_idx]
#         # label = torch.tensor(label, dtype=torch.long)  # CrossEntropyLoss expects class indices
#         # label = F.one_hot(label, num_classes=3).float()
#
#         return image, label
#
#     def __len__(self):
#         if self.augmented:
#             return len(self.df) * (augmentation_factor + 1)
#         return len(self.df)
#
#
# def create_dl(df, augmented=False):
#     return DataLoader(BreadProofingDataset(df, augmented=augmented), batch_size=batch_size, shuffle=True)
#
#
# train_dl = create_dl(train_df, False)  # MARK
# valid_dl = create_dl(valid_df)
# test_dl = create_dl(test_df)

# for i, (X, y) in enumerate(valid_dl):
#     preds = model(X)
#     print(preds)
#
# demo = torch.randn(2, 3, 256, 256)
#
# image, label = train_df.iloc[0]
# image

# for idx, (X, y) in enumerate(test_dl):
#     res = loss_fn(model(X), model(X))
#     print(res)

# print(test_dl.dataset[0])


# To visualize the dataset augmentation for a single sample
# start = (augmentation_factor + 1) * 0
# for i in range(start, start+13):
#     img, label = train_dl.dataset[i]
#     # print(type(img), type(label))
#     # print(label)
#     display(v2.functional.to_pil_image(img))

# test_loss, correct = 0,0
# for X, y in cifar10_val_dl:
#     pred = model(X)
#     test_loss += loss_fn(pred, y).item()
#     correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#

# from torchvision import datasets
# data_path = './cfar10'
#
# transforms = v2.Compose([
#     v2.ToImage(),
#     v2.ToDtype(torch.float32, scale=True),
#     v2.Resize((64, 64))
# ])
#
# cifar10 =  datasets.CIFAR10(data_path, train=True, download=True, transform=transforms)
# cifar10_val = datasets.CIFAR10(data_path, train=False, download=True, transform=transforms)
#
# cifar10_dl = DataLoader(cifar10, batch_size=batch_size, shuffle=True)
# cifar10_val_dl = DataLoader(cifar10_val, batch_size=batch_size, shuffle=True)
#
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # OR: SGD
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
#
# for t in range(epochs):
#     print(f"Epoch {t+1}\n{20*'-'}")
#     train(cifar10_dl, model, loss_fn, optimizer)
#     test(cifar10_val_dl, model, loss_fn)
#
# print(f"Completed {epochs} epochs!")
#
# test(test_dl, model, loss_fn, split="Test")

