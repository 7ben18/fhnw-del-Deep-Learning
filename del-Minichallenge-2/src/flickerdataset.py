from torch.utils.data import Dataset
import torch
from PIL import Image


class Flicker8kDataset(Dataset):
    def __init__(
        self, dataframe, image_path, image_transformations=None, caption_processor=None
    ):
        self.dataframe = dataframe
        self.image_path = image_path
        self.image_transformations = image_transformations
        self.caption_processor = caption_processor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # Get the image ID for the corresponding index
        image_id = self.dataframe.iloc[index]["image"]
        # Get the image from the image ID
        image = Image.open(self.image_path + "/" + image_id)
        # Transform the image if specified
        if self.image_transformations:
            image_pt = self.image_transformations(image)
        # Get the caption for the corresponding index
        caption = self.dataframe.iloc[index]["tokenized_caption"]
        # Convert the caption to a list of indices
        caption_idx = self.caption_processor.tokens_to_indices(caption)
        # Convert the caption to a tensor
        caption_idx_pt = torch.tensor(caption_idx)
        # Return the image and the caption
        return image_pt, caption_idx_pt
