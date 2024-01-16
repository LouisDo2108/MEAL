import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_pool
from meal.transformer import *
from meal.utils import *


class MEAL(nn.Module):
    def __init__(
        self,
        roip_output_size=(8, 8),
        dim=896,
        num_heads=8,
        bias=False,
        device="cuda",
    ):
        super().__init__()

        # Define instance variables
        self.roip_output_size = roip_output_size
        self.dim = dim
        self.num_heads = num_heads
        self.bias = bias
        self.device = device

        # Define the network layers
        self.tf = TransformerBlockV1(dim)
        self.conv11_local = nn.Conv2d(512 + 256 + 128, 448, kernel_size=1, stride=1)
        self.conv11_global_pooling = nn.Conv2d(1280, 448, kernel_size=1, stride=1)
        self.conv11_attn_feature = nn.Conv2d(1280, 896, kernel_size=1, stride=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(896, 448),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(448, 4),
        )

    def forward(self, x, debug=False):
        # Unpack the input
        small_conv, medium_conv, large_conv, bboxes, global_embed = x

        # Move the data to the GPU
        bbox = [x.to(self.device).float() for x in bboxes]
        large_conv = large_conv.to(self.device).float()
        medium_conv = medium_conv.to(self.device).float()
        small_conv = small_conv.to(self.device).float()
        global_embed = global_embed.to(self.device).float()

        # Apply ROI pooling
        x0 = roi_pool(
            small_conv,
            boxes=bbox,
            output_size=self.roip_output_size,
            spatial_scale=1 / 8.0,
        )
        x1 = roi_pool(
            medium_conv,
            boxes=bbox,
            output_size=self.roip_output_size,
            spatial_scale=1 / 16.0,
        )
        x2 = roi_pool(
            large_conv,
            boxes=bbox,
            output_size=self.roip_output_size,
            spatial_scale=1 / 32.0,
        )
        global_embedding = roi_pool(
            global_embed,
            boxes=bbox,
            output_size=self.roip_output_size,
            spatial_scale=1 / 45.0,
        )

        # Stack all local embedding together
        local_embedding = self.conv11_local(torch.cat((x0, x1, x2), dim=1))

        # Apply global and local embedding fusion (Q, K)
        global_local_embedding = torch.cat(
            (local_embedding, self.conv11_global_pooling(global_embedding)), dim=1
        )

        # Duplicate global embedding for each bounding box to the corresponding image => V
        global_embedding = torch.cat(
            [
                embedding.expand(len(b), -1, -1, -1)
                for b, embedding in zip(bbox, global_embed)
            ],
            dim=0,
        ).to(self.device)
        global_embedding = self.conv11_attn_feature(torch.tensor(global_embedding))

        # Feed Q, K, V into transformer
        x = self.tf(global_local_embedding, global_embedding)

        # GAP + FFN
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        x = self.fc(x)

        if debug:
            LOGGER.info("In debug mode, printing out shapes")
            LOGGER.info("Small Conv:  {}".format(small_conv.shape))
            LOGGER.info("Medium Conv: {}".format(medium_conv.shape))
            LOGGER.info("Large Conv:  {}".format(large_conv.shape))
            LOGGER.info("ROI small:   {}".format(x0.shape))
            LOGGER.info("ROI medium:  {}".format(x1.shape))
            LOGGER.info("ROI large:   {}".format(x2.shape))
            LOGGER.info("Local embed: {}".format(local_embedding.shape))
            LOGGER.info("Global embed:{}".format(global_embedding.shape))
            LOGGER.info("Q, K:        {}".format(global_local_embedding.shape))
            LOGGER.info("V:           {}".format(global_embedding.shape))
            LOGGER.info("Output:      {}".format(x.shape))
        return x


class MEAL_with_masked_attention(nn.Module):
    def __init__(
        self,
        dim=896,
        num_heads=8,
        num_queries=64,
        bias=False,
        device="cuda",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.bias = bias
        self.device = device

        # Define the network layers
        self.tf = TransformerBlockV1(dim)

        self.conv11_local = nn.Conv2d(512 + 256 + 128, 448, 1, 1)
        self.conv11_global_pooling = nn.Conv2d(1280, 448, 1, 1)
        self.conv11_attn_feature = nn.Conv2d(1280, 896, 1, 1)
        c2_xavier_fill(self.conv11_local)
        c2_xavier_fill(self.conv11_global_pooling)
        c2_xavier_fill(self.conv11_attn_feature)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(896, 448),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(448, 4),
        )

        self.mce0 = Mask2Former_CA(
            128,
            self.num_heads,
        )
        self.mce1 = Mask2Former_CA(
            256,
            self.num_heads,
        )
        self.mce2 = Mask2Former_CA(
            512,
            self.num_heads,
        )
        self.mce3 = Mask2Former_CA(
            1280,
            self.num_heads,
        )

        self.pe_layer0 = PositionEmbeddingSine(128 // 2, normalize=True)
        self.pe_layer1 = PositionEmbeddingSine(256 // 2, normalize=True)
        self.pe_layer2 = PositionEmbeddingSine(512 // 2, normalize=True)
        self.pe_layer3 = PositionEmbeddingSine(1280 // 2, normalize=True)
        self.query_feat0 = nn.Embedding(num_queries, 128)
        self.query_feat1 = nn.Embedding(num_queries, 256)
        self.query_feat2 = nn.Embedding(num_queries, 512)
        self.query_feat3 = nn.Embedding(num_queries, 1280)
        self.query_embed0 = nn.Embedding(num_queries, 128)
        self.query_embed1 = nn.Embedding(num_queries, 256)
        self.query_embed2 = nn.Embedding(num_queries, 512)
        self.query_embed3 = nn.Embedding(num_queries, 1280)

    def get_attn_mask_tensor(self, resize, bbox):
        # Create a list to store the masks for each bounding box
        mask_list = []

        # Iterate over each bounding box
        for box in bbox:
            # Convert the bounding box coordinates to integers and convert to a binary mask
            box = box.int().tolist()
            mask = get_binary_mask((480, 480), box).unsqueeze(0)

            # Add the mask to the list
            mask_list.append(mask)

        # Stack the masks into a tensor and resize
        mask_tensor = torch.stack(mask_list)
        mask_tensor = F.interpolate(
            mask_tensor, size=resize, mode="bilinear", align_corners=False
        )
        mask_tensor = (
            mask_tensor.flatten(2)
            .repeat(1, self.num_heads, self.num_queries, 1)
            .flatten(0, 1)
        )

        return mask_tensor

    def forward(self, x, debug=False):
        # Unpack the input
        small_conv, medium_conv, large_conv, bboxes, global_embed = x

        # Move the data to the GPU
        large_conv = large_conv.to(self.device).float()
        medium_conv = medium_conv.to(self.device).float()
        small_conv = small_conv.to(self.device).float()
        global_embed = global_embed.to(self.device).float()

        bs = len(bboxes)

        tgt0 = (
            self.query_feat0.weight.unsqueeze(1)
            .repeat(1, bs, 1)
            .to(self.device)
            .float()
        )
        tgt1 = (
            self.query_feat1.weight.unsqueeze(1)
            .repeat(1, bs, 1)
            .to(self.device)
            .float()
        )
        tgt2 = (
            self.query_feat2.weight.unsqueeze(1)
            .repeat(1, bs, 1)
            .to(self.device)
            .float()
        )
        tgt3 = (
            self.query_feat3.weight.unsqueeze(1)
            .repeat(1, bs, 1)
            .to(self.device)
            .float()
        )

        memory0 = small_conv.flatten(2).permute(2, 0, 1)
        memory1 = medium_conv.flatten(2).permute(2, 0, 1)
        memory2 = large_conv.flatten(2).permute(2, 0, 1)
        memory3 = global_embed.flatten(2).permute(2, 0, 1)

        attn_mask_tensor0 = (
            self.get_attn_mask_tensor(small_conv.shape[-2:], bboxes)
            .to(self.device)
            .float()
            .detach()
        )
        attn_mask_tensor1 = (
            self.get_attn_mask_tensor(medium_conv.shape[-2:], bboxes)
            .to(self.device)
            .float()
            .detach()
        )
        attn_mask_tensor2 = (
            self.get_attn_mask_tensor(large_conv.shape[-2:], bboxes)
            .to(self.device)
            .float()
            .detach()
        )
        attn_mask_tensor3 = (
            self.get_attn_mask_tensor(global_embed.shape[-2:], bboxes)
            .to(self.device)
            .float()
            .detach()
        )

        query_embed0 = (
            self.query_embed0.weight.unsqueeze(1)
            .repeat(1, bs, 1)
            .to(self.device)
            .float()
        )
        query_embed1 = (
            self.query_embed1.weight.unsqueeze(1)
            .repeat(1, bs, 1)
            .to(self.device)
            .float()
        )
        query_embed2 = (
            self.query_embed2.weight.unsqueeze(1)
            .repeat(1, bs, 1)
            .to(self.device)
            .float()
        )
        query_embed3 = (
            self.query_embed3.weight.unsqueeze(1)
            .repeat(1, bs, 1)
            .to(self.device)
            .float()
        )

        pos0 = (
            self.pe_layer0(small_conv, None)
            .flatten(2)
            .permute(2, 0, 1)
            .to(self.device)
            .float()
        )
        pos1 = (
            self.pe_layer1(medium_conv, None)
            .flatten(2)
            .permute(2, 0, 1)
            .to(self.device)
            .float()
        )
        pos2 = (
            self.pe_layer2(large_conv, None)
            .flatten(2)
            .permute(2, 0, 1)
            .to(self.device)
            .float()
        )
        pos3 = (
            self.pe_layer3(global_embed, None)
            .flatten(2)
            .permute(2, 0, 1)
            .to(self.device)
            .float()
        )

        # Apply masked attention
        x0 = self.mce0(
            tgt=tgt0,
            memory=memory0,
            memory_mask=attn_mask_tensor0,
            memory_key_padding_mask=None,  # here we do not apply masking on padded region
            pos=pos0,
            query_pos=query_embed0,
        )

        x1 = self.mce1(
            tgt=tgt1,
            memory=memory1,
            memory_mask=attn_mask_tensor1,
            memory_key_padding_mask=None,  # here we do not apply masking on padded region
            pos=pos1,
            query_pos=query_embed1,
        )

        x2 = self.mce2(
            tgt=tgt2,
            memory=memory2,
            memory_mask=attn_mask_tensor2,
            memory_key_padding_mask=None,  # here we do not apply masking on padded region
            pos=pos2,
            query_pos=query_embed2,
        )

        global_embedding = self.mce3(
            tgt=tgt3,
            memory=memory3,
            memory_mask=attn_mask_tensor3,
            memory_key_padding_mask=None,  # here we do not apply masking on padded region
            pos=pos3,
            query_pos=query_embed3,
        )

        x0 = x0.permute(1, 2, 0)
        x1 = x1.permute(1, 2, 0)
        x2 = x2.permute(1, 2, 0)
        local_embedding = self.conv11_local(
            torch.cat((x0, x1, x2), dim=1).reshape(-1, 896, 8, 8)
        )
        global_embedding = global_embedding.permute(1, 2, 0).reshape(-1, 1280, 8, 8)

        # Apply global and local feature fusion (Q, K)
        global_local_embedding = torch.cat(
            (local_embedding, self.conv11_global_pooling(global_embedding)), dim=1
        )
        global_embedding = self.conv11_attn_feature(global_embedding)  # V

        # Feed Q, K, V into transformer
        x = self.tf(global_local_embedding, global_embedding)

        # GAP + FFN
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        x = self.fc(x)

        if debug:
            LOGGER.info("In debug mode, printing out shapes")
            LOGGER.info("Small Conv:   {}".format(small_conv.shape))
            LOGGER.info("Medium Conv:  {}".format(medium_conv.shape))
            LOGGER.info("Large Conv:   {}".format(large_conv.shape))
            LOGGER.info("M-attn small: {}".format(x0.shape))
            LOGGER.info("M-attn medium:{}".format(x1.shape))
            LOGGER.info("M-attn large: {}".format(x2.shape))
            LOGGER.info("Local embed:  {}".format(local_embedding.shape))
            LOGGER.info("Global embed: {}".format(global_embedding.shape))
            LOGGER.info("Q, K:         {}".format(global_local_embedding.shape))
            LOGGER.info("V:            {}".format(global_embedding.shape))
            LOGGER.info("Output:       {}".format(x.shape))
        return x


def init_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "MEAL":
        model = MEAL(device=device)
    elif model_name == "MEAL_with_masked_attention":
        model = MEAL_with_masked_attention(device=device)

    loss_func = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 1.0, 1.5, 1.5]).to(device), label_smoothing=0.1
    )
    opt = torch.optim.AdamW(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=100, eta_min=1e-6
    )
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     opt, mode="min", factor=0.5, patience=10, verbose=1
    # )
    return model, loss_func, opt, lr_scheduler
