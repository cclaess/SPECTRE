"""
Implementation of the CLIP framework for text-image feature alignment.

This module provides the necessary components to train the CLIP framework an is based on the 
original paper: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021), 
https://arxiv.org/abs/2103.00020

Addional resources:
Hamamci et al., "Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography" (2024),
https://arxiv.org/abs/2403.17834
"""
import math
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from spectre.models.vits import VisionTransformer, FeatureVisionTransformer
from spectre.models.text_encoders import GeneralTextEncoder


class CLIP(nn.Module):
    def __init__(
            self,
            image_backbone: nn.Module,
            text_backbone: nn.Module,
            grid_size: Tuple[int, int, int],
    ):
        super().__init__()
        self.image_backbone = image_backbone
        self.text_backbone = text_backbone
        self.grid_size = grid_size
    
    def forward(self, images, texts):
        image_features = self.image_backbone(images)

        # MONAI dataloaders will push chunks of the same CT volume to the batch dimension
        # Reorder batch so that chunks of the same CT volume are together
        B, C  = image_features.shape
        B_eff = B // math.prod(self.grid_size)
        image_features = image_features.view(B_eff, -1, C)

        # Max-pool over chunks of the same CT volume
        image_features = image_features.max(dim=1).values

        text_features = self.text_backbone(texts)
        return image_features, text_features


class SigLIP3D(nn.Module):
    def __init__(self,
                 
        # Vision encoder parameters
        vision_encoder: nn.Module = VisionTransformer(),
        vision_feature_comb: nn.Module = FeatureVisionTransformer(),

        # Text encoder parameters
        text_encoder: nn.Module = GeneralTextEncoder(),
        # Common embedding dimension and logit scale initialization
        embed_dim: int = 768,
        logit_scale_init: float = 2.6592
    ):
        """
        3D SigLIP: An improved CLIP model combining a novel 3D Vision Transformer and an improved text encoder.
        """
        super().__init__()
        # Initialize vision encoder
        self.vision_encoder = vision_encoder
        self.vision_feature_comb = vision_feature_comb

        # Initialize text encoder 
        self.text_encoder = text_encoder

        self.embed_dim = embed_dim
        # Learnable logit scale parameter for contrastive similarity
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)


    def forward(self, images: torch.Tensor, tokens: torch.Tensor):
        """
        Args:
            images: Tensor of shape (batch, crops, height, width, depth)
            tokens: Tensor of shape (batch, text_context_length)
        Returns:
            logits_per_image: Tensor (batch, batch) with similarity scores.
            logits_per_text: Tensor (batch, batch) with similarity scores.
        """

        # reshape input to be (batch x crops, 1,  height, width, depth)
        B, C, H, W, D = images.shape
        images = images.view(B*C, 1, H, W, D)

        # Compute embeddings for both modalities.
        crop_embeddings = self.vision_encoder(images, pre_logits=True) # (batch, crops, embed_dim)
        print(crop_embeddings.shape)
        crop_embeddings = crop_embeddings.view(B, C, -1)
        print(crop_embeddings.shape)
        image_embeddings = self.vision_feature_comb(crop_embeddings) # (batch, embed_dim)

        text_embeddings = self.text_encoder(tokens) # (batch, embed_dim)

        # Normalize the embeddings.
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        # print(image_embeddings.shape, text_embeddings.shape)

        # # Compute similarity logits.
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_embeddings @ text_embeddings.t()
        # logits_per_text = logits_per_image.t()

        return image_embeddings, text_embeddings
    

# Example usage
if __name__ == "__main__":

    # Initialize the SigLIP model
    vision_encoder = VisionTransformer()
    vision_feature_comb = FeatureVisionTransformer(patch_dim=768, num_patches=2)
    text_encoder = GeneralTextEncoder()
    model = SigLIP3D(vision_encoder, vision_feature_comb, text_encoder, embed_dim=768)

    # Generate random input data
    images = torch.randn(2, 2, 128, 128, 64)

    text_prompts = ["There is no pneumothorax or pleural effusion",
                "No pleural effusion or pneumothorax is seen"]
    
    url = "microsoft/BiomedVLP-CXR-BERT-specialized"
    tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
    # tokens = torch.randint(0, 49152, (2, 768))

    tokenizer_output = tokenizer.batch_encode_plus(batch_text_or_text_pairs=text_prompts,
                                               add_special_tokens=True,
                                               padding='longest',
                                               return_tensors='pt')
    print(tokenizer_output['input_ids'].shape)

    # Forward pass
    logits_per_image, logits_per_text = model(images, tokenizer_output['input_ids'])
    print(logits_per_image.shape, logits_per_text.shape)
