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
from typing import Tuple, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# import sys
# sys.path.append(r"C:\Users\20195435\OneDrive - TU Eindhoven\TUe\Projects\SPECTRE")

from spectre.models.vits import VisionTransformer, FeatureVisionTransformer
from spectre.models.text_encoders import GeneralTextEncoder
from spectre.models.qwen_text_encoders import Qwen2Model, Qwen2Config
from spectre.models.tokenization_qwen import Qwen2Tokenizer
from safetensors.torch import load_file

from spectre.ssl.losses import siglip_loss

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
        pretrained_vision_encoder_path: str = "weights/clip/vit_base_patch16_224.pth",
        vision_feature_comb: nn.Module = FeatureVisionTransformer(),
        pretrained_vision_feature_comb_path: str = "weights/clip/vit_base_patch16_224.pth",

        # Text encoder parameters
        text_encoder: nn.Module = Qwen2Model(Qwen2Config.from_pretrained("infgrad/jasper_en_vision_language_v1", is_text_encoder=True)),
        pretrained_text_encoder_path: str = "weights/clip/jasper_en_vision_language_v1_base.pth",
        # Common embedding dimension and logit scale initialization
        embed_dim: int = 768
    ):
        """
        3D SigLIP: An improved CLIP model combining a novel 3D Vision Transformer and an improved text encoder.
        """
        super().__init__()
        # Initialize vision encoder
        self.vision_encoder = vision_encoder
        if pretrained_vision_encoder_path:
            self.vision_encoder.load_state_dict(torch.load(pretrained_vision_encoder_path))
        self.vision_feature_comb = vision_feature_comb
        if pretrained_vision_feature_comb_path:
            self.vision_feature_comb.load_state_dict(torch.load(pretrained_vision_feature_comb_path))
        
        # Initialize text encoder 
        self.text_encoder = text_encoder
        
        self.pretrained_text_encoder_path = pretrained_text_encoder_path
        if pretrained_text_encoder_path:
            self.text_encoder.load_state_dict(torch.load(pretrained_text_encoder_path))

        # self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((self.config.num_img_tokens, config.text_config.hidden_size))

        self.embed_dim = embed_dim
        self.vector_linear_vision = nn.Linear(embed_dim, embed_dim, bias=True)
        self.vector_linear_text = nn.Linear(1536, embed_dim, bias=True)

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))


    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, return_loss: bool = False):
        """
        Args:
            images: Tensor of shape (batch, crops, height, width, depth)
            tokens: Tensor of shape (batch, sequence_length)
            attention_mask: Tensor of shape (batch, sequence_length)

        Returns:
            logits_per_image: Tensor (batch, batch) with similarity scores.
            logits_per_text: Tensor (batch, batch) with similarity scores.
        """

        # reshape input to be (batch x crops, 1,  height, width, depth)
        B, C, H, W, D = images.shape
        images = images.view(B*C, 1, H, W, D)

        # Compute embeddings for both modalities.
        crop_embeddings = self.vision_encoder(images, pre_logits=True) # (batch, crops, embed_dim)
        crop_embeddings = crop_embeddings.view(B, C, -1)
        image_embeddings = self.vision_feature_comb(crop_embeddings) # (batch, embed_dim)
        print(image_embeddings.shape)   
        # Compute text embeddings
        text_encoding = text_encoder(input_ids=text_tokens,
                                attention_mask=attention_mask)
        
        last_hidden = text_encoding.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        text_embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        print(text_embeddings.shape)

        # Project the embeddings to the same dimension.
        image_embeddings = self.vector_linear_vision(image_embeddings)
        text_embeddings = self.vector_linear_text(text_embeddings)


        image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeddings, image_embeddings.t().to(text_embeddings.device))

        logit_scale, logit_bias = self.logit_scale.to(text_embeddings.device), self.logit_bias.to(text_embeddings.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            # Adapted from https://github.com/google-research/big_vision/blob/01edb81a4716f93a48be43b3a4af14e29cdb3a7f/big_vision/trainers/proj/image_text/siglip.py#L287
            eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()

            return loss      

        return image_embeddings, text_embeddings, logits_per_image, logits_per_text
    

    

# Example usage
if __name__ == "__main__":

    # Initialize the SigLIP model
    vision_encoder = VisionTransformer()
    vision_feature_comb = FeatureVisionTransformer(patch_dim=768, num_patches=2)
    model_config = Qwen2Config.from_pretrained("infgrad/jasper_en_vision_language_v1", cache_dir=r"C:\Users\20195435\OneDrive - TU Eindhoven\TUe\Projects\SPECTRE\temp")
    print(model_config)
    text_encoder = Qwen2Model(model_config)

    # load safe tensor
    loaded = load_file(r"C:\Users\20195435\.cache\huggingface\hub\models--infgrad--jasper_en_vision_language_v1\snapshots\6fae668ae57688bf9c54e02e86dfc4d7403881bb\model.safetensors", device='cpu')
    print(loaded.keys())

    loaded = {k: v for k, v in loaded.items() if "vision" not in k}
    # rename some of the keys "model" to ""
    loaded = {k.replace("model.", ""): v for k, v in loaded.items()}
    # remove keys with the word "vision" in them
    
    text_encoder.load_state_dict(loaded, strict=False)

    # save the model
    torch.save(text_encoder.state_dict(), r"C:\Users\20195435\OneDrive - TU Eindhoven\TUe\Projects\SPECTRE\temp\jasper_en_vision_language_v1_base.pth")
    print(text_encoder)

    qwen_config = text_encoder.config
    print(qwen_config)

    model = SigLIP3D(vision_encoder, None, vision_feature_comb, None, text_encoder, None, embed_dim=768)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of SigLIP3D parameters: {total_params}")


    # Generate random input data
    images = torch.randn(2, 2, 128, 128, 64)

    text_prompts = ["There is no pneumothorax or pleural effusion",
                "No pleural effusion or pneumothorax is seen"]
    
    # data
    text_prompts = [
        "s2p_query: Why the sky is blue?",
        "s2p_query: how to choose suitable color",
    ]
    
    tokenizer = Qwen2Tokenizer.from_pretrained("infgrad/jasper_en_vision_language_v1", cache_dir=r"C:\Users\20195435\OneDrive - TU Eindhoven\TUe\Projects\SPECTRE\temp")
    tokenizer_output = tokenizer.batch_encode_plus(batch_text_or_text_pairs=text_prompts,
                                               add_special_tokens=True,
                                               padding='longest',
                                               return_tensors='pt')

    total_params = sum(p.numel() for p in text_encoder.parameters())
    print(f"Number of text_encoder parameters: {total_params}")

    # Forward pass
    loss = model.forward(images, tokenizer_output['input_ids'], tokenizer_output['attention_mask'], return_loss=True)
    print(loss)
    # Forward pass
    image_embeddings, text_embeddings, logits_per_image, logits_per_text = model.forward(images, tokenizer_output['input_ids'], tokenizer_output['attention_mask'])

    print(image_embeddings.shape, text_embeddings.shape, logits_per_image.shape, logits_per_text.shape)
