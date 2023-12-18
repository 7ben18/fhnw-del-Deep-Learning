import os
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class ImageCaptionDataset(Dataset):
    def __init__(
        self,
        dataframe,
        image_path,
        vocab_size,
        caption_processor,
        transform=None,
        max_length=20,
    ):
        self.dataframe = dataframe
        self.images_path = image_path
        self.vocab_size = vocab_size
        self.caption_processor = caption_processor
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_name = self.dataframe.iloc[idx]["image"]
        image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption = self.dataframe.iloc[idx]["caption"]
        caption_tokens = self.caption_processor.caption_to_tokens(caption)
        caption_tokens = caption_tokens[: self.max_length]

        # Pad the caption tokens to the maximum length or trim
        if len(caption_tokens) < self.max_length:
            # padding
            caption_tokens = caption_tokens + [
                self.caption_processor.word_to_index["<PAD>"]
            ] * (self.max_length - len(caption_tokens))
        else:
            # trimming
            caption_tokens = caption_tokens[: self.max_length]

        caption_tokens = torch.tensor(caption_tokens, dtype=torch.long)

        return image, caption_tokens


class DataPreparation:
    def __init__(
        self, image_path, vocab_size, caption_info, image_transformations, batch_size=64
    ):
        self.image_path = image_path
        self.vocab_size = vocab_size
        self.caption_info = caption_info
        self.image_transformations = image_transformations
        self.batch_size = batch_size

    def split_dataframe_by_images(
        self, dataframe, val_test_size=0.4, random_state=42, get_len=False
    ):
        unique_images = dataframe["image"].unique()

        train_images, val_test_images = train_test_split(
            unique_images, test_size=val_test_size, random_state=random_state
        )
        val_images, test_images = train_test_split(
            val_test_images, test_size=0.5, random_state=random_state
        )

        train_df = dataframe[dataframe["image"].isin(train_images)]
        val_df = dataframe[dataframe["image"].isin(val_images)]
        test_df = dataframe[dataframe["image"].isin(test_images)]

        if get_len:
            print("Overview of length after split:")
            print("Number of unique images:", len(unique_images))
            print(f"Train dataset contains {len(train_df)} items.")
            print(f"Validation dataset contains {len(val_df)} items.")
            print(f"Test dataset contains {len(test_df)} items.")

        return train_df, val_df, test_df

    def create_datasets(self, dataframe, get_len=False):
        train_df, val_df, test_df = self.split_dataframe_by_images(
            dataframe, get_len=get_len
        )
        train_dataset = ImageCaptionDataset(
            train_df,
            self.image_path,
            self.vocab_size,
            self.caption_info,
            self.image_transformations,
        )
        val_dataset = ImageCaptionDataset(
            val_df,
            self.image_path,
            self.vocab_size,
            self.caption_info,
            self.image_transformations,
        )
        test_dataset = ImageCaptionDataset(
            test_df,
            self.image_path,
            self.vocab_size,
            self.caption_info,
            self.image_transformations,
        )

        if get_len:
            train_len, val_len, test_len = (
                len(train_dataset),
                len(val_dataset),
                len(test_dataset),
            )
            print("Overview of items in Dataset:")
            print(f"Train dataset contains {train_len} items.")
            print(f"Validation dataset contains {val_len} items.")
            print(f"Test dataset contains {test_len} items.")

        return train_dataset, val_dataset, test_dataset

    def create_data_loaders(self, dataframe, get_len=False):
        train_dataset, val_dataset, test_dataset = self.create_datasets(
            dataframe, get_len=get_len
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=2,
            persistent_workers=False,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=2,
            persistent_workers=False,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=2,
            persistent_workers=False,
            shuffle=False,
        )

        if get_len:
            train_len, val_len, test_len = (
                len(train_loader),
                len(val_loader),
                len(test_loader),
            )
            print("Overview of batches:")
            print(f"Number of batches in train_loader: {train_len}")
            print(f"Number of batches in val_loader: {val_len}")
            print(f"Number of batches in test_loader: {test_len}")

        return train_loader, val_loader, test_loader
