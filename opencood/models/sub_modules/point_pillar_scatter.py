import os

import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['num_features']
        self.nx, self.ny, self.nz = model_cfg['grid_size']  # [704, 200, 1] 

        assert self.nz == 1
        self.strict_empty_voxels = str(
            os.environ.get("STRICT_EMPTY_VOXELS", "0")
        ).lower() in ("1", "true", "yes")
        self.warn_empty_voxels = str(
            os.environ.get("WARN_EMPTY_VOXELS", "1")
        ).lower() in ("1", "true", "yes")
        self.warn_every = max(1, int(os.environ.get("WARN_EMPTY_VOXELS_EVERY", "50")))
        self._empty_event_count = 0

    def _handle_empty_voxel_event(self, message: str) -> None:
        self._empty_event_count += 1
        if self.strict_empty_voxels:
            raise RuntimeError(message)
        if self.warn_empty_voxels and (
            self._empty_event_count == 1
            or self._empty_event_count % self.warn_every == 0
        ):
            print(
                "[PointPillarScatter][WARN] "
                f"{message} (event={self._empty_event_count})"
            )

    def forward(self, batch_dict):
        """ 将生成的pillar按照坐标索引还原到原空间中
        Args:
            pillar_features:(M, 64)
            coords:(M, 4) 第一维是batch_index

        Returns:
            batch_spatial_features:(4, 64, H, W)
            
            |-------|
            |       |             |-------------|
            |       |     ->      |  *          |
            |       |             |             |
            | *     |             |-------------|
            |-------|

            Lidar Point Cloud        Feature Map
            x-axis up                Along with W 
            y-axis right             Along with H

            Something like clockwise rotation of 90 degree.

        """
        pillar_features = batch_dict['pillar_features']
        coords = batch_dict['voxel_coords']
        record_len = batch_dict.get('record_len')

        if record_len is not None:
            batch_size = int(record_len.sum().item())
        elif coords.numel() > 0:
            batch_size = coords[:, 0].max().int().item() + 1
        else:
            batch_size = 0
            self._handle_empty_voxel_event(
                "All voxel coords are empty and record_len is missing; "
                "producing an empty BEV batch."
            )

        batch_spatial_features = torch.zeros(
            batch_size,
            self.num_bev_features,
            self.nz * self.nx * self.ny,
            dtype=pillar_features.dtype,
            device=pillar_features.device)

        empty_batch_indices = []
        for batch_idx in range(batch_size):
            # batch_index的mask
            batch_mask = coords[:, 0] == batch_idx
            if not batch_mask.any().item():
                empty_batch_indices.append(batch_idx)
                continue
            # 根据mask提取坐标
            this_coords = coords[batch_mask, :] # (batch_idx_voxel,4)  # zyx order, x in [0,706], y in [0,200]
            # 这里的坐标是b,z,y和x的形式,且只有一层，因此计算索引的方式如下
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            # 转换数据类型
            indices = indices.type(torch.long)
            # 根据mask提取pillar_features
            pillars = pillar_features[batch_mask, :] # (batch_idx_voxel,64)
            pillars = pillars.t() # (64,batch_idx_voxel)
            # 在索引位置填充pillars
            batch_spatial_features[batch_idx, :, indices] = pillars

        if empty_batch_indices:
            record_len_info = (
                record_len.detach().cpu().tolist() if record_len is not None else None
            )
            self._handle_empty_voxel_event(
                "Detected empty voxel inputs for some batch entries: "
                f"empty_indices={empty_batch_indices}, "
                f"batch_size={batch_size}, "
                f"record_len={record_len_info}, "
                f"coords_shape={tuple(coords.shape)}"
            )

        batch_spatial_features = batch_spatial_features.view(
            batch_size,
            self.num_bev_features * self.nz,
            self.ny,
            self.nx) # It put y axis(in lidar frame) as image height. [..., 200, 704]
        batch_dict['spatial_features'] = batch_spatial_features

        return batch_dict
