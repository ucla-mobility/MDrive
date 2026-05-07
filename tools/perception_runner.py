"""
PerceptionRunner — reusable wrapper around HEAL HeterBaseline detectors.

Encapsulates the load-checkpoint / voxelize / forward / decode pipeline.
Two modes:

    runner.detect(pcd_np)
        Single-agent forward pass — degenerate (the cross-agent fusion layer
        self-fuses with one input). Useful for debugging only.

    runner.detect_cooperative(lidars, agent_poses_world, primary_idx=0)
        Multi-agent cooperative forward pass — what these models were
        designed for. Each agent's LiDAR is voxelized separately, stacked
        into one batch, and the fusion layer warps cooperators' features
        into the primary's BEV before fusion. Returns detections in the
        primary agent's LiDAR frame.

LiDAR modality only. m2 (camera) path is dead code here.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.heter_model_baseline import HeterModelBaseline
from opencood.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.utils.transformation_utils import x_to_world, x1_to_x2


# Map a friendly detector name to its checkpoint dir + best-val .pth filename.
# Add CoBEVT here once the SwapFusion port lands.
# All four detectors now point at the OPV2V `lidar` HuggingFace variant
# (HeterBaseline_opv2v_lidar_<det>_*.zip from yifanlu/HEAL). Heads were
# trained without expecting cameras, so feeding LiDAR-only matches their
# training distribution and gives ~+7 pts AP@.50 on OPV2V test vs the
# `lidarcamera` variants. The original lidarcamera ckpts are still on
# disk under ckpt/heter_baseline/<det>/ if needed for an ablation.
_DETECTORS = {
    "attfuse": ("ckpt/heter_baseline/attfuse_lidaronly", "net_epoch_bestval_at15.pth"),
    "fcooper": ("ckpt/heter_baseline/fcooper_lidaronly", "net_epoch_bestval_at23.pth"),
    "disco":   ("ckpt/heter_baseline/disco_lidaronly",   "net_epoch_bestval_at23.pth"),
    "cobevt":  ("ckpt/heter_baseline/cobevt_lidaronly",  "net_epoch_bestval_at39.pth"),
    # Aliases preserved so the openloop-gated auto-switch remains a no-op
    # rather than an error if anything still references the _lidar suffix.
    "attfuse_lidar": ("ckpt/heter_baseline/attfuse_lidaronly", "net_epoch_bestval_at15.pth"),
    "fcooper_lidar": ("ckpt/heter_baseline/fcooper_lidaronly", "net_epoch_bestval_at23.pth"),
    "disco_lidar":   ("ckpt/heter_baseline/disco_lidaronly",   "net_epoch_bestval_at23.pth"),
    "cobevt_lidar":  ("ckpt/heter_baseline/cobevt_lidaronly",  "net_epoch_bestval_at39.pth"),
}


@dataclass
class Detections:
    """Output of one detector forward pass.

    All coordinates are in the ego (LiDAR sensor) frame, +x forward.
    """
    corners:    np.ndarray   # (P, 8, 3)  the 8 box corners in xyz
    scores:     np.ndarray   # (P,)
    centers_xy: np.ndarray   # (P, 2) — centroid of the 8 corners, xy only
    P:          int          # number of detections (after NMS)


class PerceptionRunner:
    def __init__(
        self,
        detector: str = "fcooper",
        device: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ):
        # Openloop runs (--openloop sets CUSTOM_EGO_LOG_REPLAY=1) feed only
        # LiDAR to the model. The default lidarcamera checkpoints have heads
        # trained on m1+m2 fused features and lose ~7pts AP@.50 when starved
        # of camera input. Auto-prefer the lidaronly variant in openloop if
        # one is registered (e.g. "fcooper" → "fcooper_lidar"). Closed-loop
        # behavior unchanged. Override with PERCEPTION_SWAP_FORCE_DETECTOR=...
        if (os.environ.get("CUSTOM_EGO_LOG_REPLAY", "0") == "1"
                and not os.environ.get("PERCEPTION_SWAP_FORCE_DETECTOR")):
            lidar_alias = f"{detector}_lidar"
            if lidar_alias in _DETECTORS:
                print(f"[perception_runner] openloop detected → using "
                      f"{lidar_alias} (lidar-only ckpt) instead of {detector}")
                detector = lidar_alias

        if detector not in _DETECTORS:
            raise ValueError(
                f"unknown detector {detector!r}; available: {sorted(_DETECTORS)}"
            )
        ckpt_subdir, weight_name = _DETECTORS[detector]
        ckpt_dir     = os.path.join(REPO_ROOT, ckpt_subdir)
        config_path  = os.path.join(ckpt_dir, "config.yaml")
        weights_path = os.path.join(ckpt_dir, weight_name)

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"missing config: {config_path}")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"missing weights: {weights_path}")

        self.detector_name = detector
        self.config        = load_yaml(config_path)
        model_args         = self.config["model"]["args"]

        # Build the model and load weights.
        self.model = HeterModelBaseline(model_args)
        state = torch.load(weights_path, map_location="cpu")
        load_result = self.model.load_state_dict(state, strict=False)
        # We don't print here — it's library code. The smoke test prints these.
        self._missing    = load_result.missing_keys
        self._unexpected = load_result.unexpected_keys

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()

        # Voxelizer params live under heter.modality_setting.m1.preprocess —
        # the top-level `preprocess:` block is a no-op sentinel (max_voxel=1).
        m1_pre = self.config["heter"]["modality_setting"]["m1"]["preprocess"]
        self.preprocessor = SpVoxelPreprocessor(m1_pre, train=False)

        # Postprocess params (anchor box, NMS thresh, score thresh).
        post_params = dict(self.config["postprocess"])
        if score_threshold is not None:
            post_params = {**post_params}
            post_params["target_args"] = {
                **post_params["target_args"],
                "score_threshold": float(score_threshold),
            }
        self.postprocessor = VoxelPostprocessor(post_params, train=False)
        self._anchor_box = torch.from_numpy(
            self.postprocessor.generate_anchor_box()
        ).to(self.device)

        self._max_cav = int(self.config["train_params"]["max_cav"])
        # Cache m2 (camera) preprocessing config if the model has it. Only
        # used by detect_multimodal(); harmless to leave attached otherwise.
        try:
            self._m2_cfg = self.config["heter"]["modality_setting"]["m2"]
            self._has_m2 = True
        except (KeyError, TypeError):
            self._m2_cfg = None
            self._has_m2 = False

        # OPV2V test pcds don't ship depth maps; disable depth_supervision so
        # the m2 encoder doesn't try to read a 4th depth channel from the
        # input image (lss_submodule.py:122 indexes x[:, 3, :, :] when
        # depth_supervision is True). Inference uses the predicted depth path.
        if self._has_m2 and hasattr(self.model, "encoder_m2"):
            enc_m2 = self.model.encoder_m2
            if hasattr(enc_m2, "depth_supervision"):
                enc_m2.depth_supervision = False
            if hasattr(enc_m2, "camencode"):
                ce = enc_m2.camencode
                if hasattr(ce, "depth_supervision"):
                    ce.depth_supervision = False
                if hasattr(ce, "use_gt_depth"):
                    ce.use_gt_depth = False
        # Model-level supervision flag must also be off; otherwise
        # heter_model_baseline.py:188 tries to read encoder.depth_items
        # which we no longer populate.
        if self._has_m2 and hasattr(self.model, "depth_supervision_m2"):
            self.model.depth_supervision_m2 = False

    @torch.no_grad()
    def detect_multimodal(self, pcd_np: np.ndarray, m2_inputs: dict) -> "Detections":
        """Run a `lidarcamera` HeterBaseline checkpoint with both modalities
        for a single ego CAV.

        m2_inputs must contain (already-preprocessed, on CPU as torch tensors):
          - imgs       (Ncam, 3, fH, fW)  — normalize_img'd RGB
          - intrins    (Ncam, 3, 3)
          - rots       (Ncam, 3, 3)        — R_lidar_camera (3x3)
          - trans      (Ncam, 3)           — T_lidar_camera (3,)
          - extrinsics (Ncam, 4, 4)        — full T_lidar_camera
          - post_rots  (Ncam, 3, 3)        — image-augmentation rotation
          - post_trans (Ncam, 3)           — image-augmentation translation

        See `intermediate_heter_fusion_dataset.py:177-241` for the canonical
        builder. The model treats m1 and m2 of the same CAV as TWO virtual
        agent slots in record_len, so we set agent_modality_list=["m1","m2"]
        and record_len=[2]."""
        if not self._has_m2:
            raise RuntimeError("This checkpoint config has no m2 modality "
                               "setting; use detect() / detect_cooperative() instead.")

        if pcd_np.dtype != np.float32:
            pcd_np = pcd_np.astype(np.float32, copy=False)
        v = self.preprocessor.preprocess(pcd_np)
        coords = np.concatenate(
            [np.zeros((v["voxel_coords"].shape[0], 1), dtype=v["voxel_coords"].dtype),
             v["voxel_coords"]],
            axis=1,
        )
        inputs_m1 = {
            "voxel_features":   torch.from_numpy(v["voxel_features"]).float().to(self.device),
            "voxel_coords":     torch.from_numpy(coords).int().to(self.device),
            "voxel_num_points": torch.from_numpy(v["voxel_num_points"]).int().to(self.device),
        }
        # m2 model expects (B, Ncam, 3, H, W) — add the batch dim (B=1).
        inputs_m2 = {k: v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v
                     for k, v in m2_inputs.items()}

        data_dict = {
            "agent_modality_list": ["m1", "m2"],
            "record_len":          torch.tensor([2], dtype=torch.int).to(self.device),
            # 2 virtual slots, identity transform between them (same physical CAV).
            "pairwise_t_matrix":   torch.eye(4).view(1, 1, 1, 4, 4)
                                          .repeat(1, self._max_cav, self._max_cav, 1, 1).to(self.device),
            "inputs_m1":           inputs_m1,
            "inputs_m2":           inputs_m2,
        }

        out = self.model(data_dict)

        proc_in = {
            "ego": {
                "transformation_matrix": torch.eye(4).to(self.device),
                "anchor_box":            self._anchor_box,
            }
        }
        proc_out = {"ego": {k: v for k, v in out.items()
                            if k in ("cls_preds", "reg_preds", "dir_preds")}}
        pred_boxes, pred_scores = self.postprocessor.post_process(proc_in, proc_out)
        if pred_boxes is None or len(pred_boxes) == 0:
            return Detections(
                corners=np.zeros((0, 8, 3), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                centers_xy=np.zeros((0, 2), dtype=np.float32),
                P=0,
            )
        corners = pred_boxes.cpu().numpy()
        scores = pred_scores.cpu().numpy()
        centers_xy = corners.mean(axis=1)[:, :2]
        return Detections(corners=corners.astype(np.float32),
                          scores=scores.astype(np.float32),
                          centers_xy=centers_xy.astype(np.float32),
                          P=int(corners.shape[0]))

    @torch.no_grad()
    def detect(self, pcd_np: np.ndarray) -> Detections:
        """Run the detector on one ego-frame LiDAR cloud.

        Parameters
        ----------
        pcd_np : (N, 4) float32 — x, y, z, intensity in the ego LiDAR frame.

        Returns
        -------
        Detections in the ego frame.
        """
        if pcd_np.dtype != np.float32:
            pcd_np = pcd_np.astype(np.float32, copy=False)
        if pcd_np.ndim != 2 or pcd_np.shape[1] != 4:
            raise ValueError(f"pcd_np must be (N, 4); got {pcd_np.shape}")

        # 1. Voxelize.
        v = self.preprocessor.preprocess(pcd_np)
        # PointPillar's scatter expects voxel_coords with a leading batch
        # index column (b, z, y, x). For batch size 1, prepend zeros.
        coords = np.concatenate(
            [np.zeros((v["voxel_coords"].shape[0], 1), dtype=v["voxel_coords"].dtype),
             v["voxel_coords"]],
            axis=1,
        )

        inputs_m1 = {
            "voxel_features":   torch.from_numpy(v["voxel_features"]).float().to(self.device),
            "voxel_coords":     torch.from_numpy(coords).int().to(self.device),
            "voxel_num_points": torch.from_numpy(v["voxel_num_points"]).int().to(self.device),
        }

        # 2. Build the data_dict the model consumes for a single-agent batch.
        data_dict = {
            "agent_modality_list": ["m1"],
            "record_len":          torch.tensor([1], dtype=torch.int).to(self.device),
            "pairwise_t_matrix":   torch.eye(4).view(1, 1, 1, 4, 4)
                                          .repeat(1, self._max_cav, self._max_cav, 1, 1).to(self.device),
            "inputs_m1":           inputs_m1,
        }

        # 3. Forward pass.
        out = self.model(data_dict)

        # 4. Decode heads → 8-corner boxes via the trained postprocessor.
        proc_in = {
            "ego": {
                "transformation_matrix": torch.eye(4).to(self.device),
                "anchor_box":            self._anchor_box,
            }
        }
        proc_out = {"ego": {k: v for k, v in out.items()
                            if k in ("cls_preds", "reg_preds", "dir_preds")}}
        pred_boxes, pred_scores = self.postprocessor.post_process(proc_in, proc_out)

        if pred_boxes is None or len(pred_boxes) == 0:
            return Detections(
                corners=np.zeros((0, 8, 3), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                centers_xy=np.zeros((0, 2), dtype=np.float32),
                P=0,
            )

        corners    = pred_boxes.cpu().numpy()
        scores     = pred_scores.cpu().numpy()
        centers_xy = corners.mean(axis=1)[:, :2]
        return Detections(corners=corners, scores=scores,
                          centers_xy=centers_xy, P=len(scores))

    @torch.no_grad()
    def detect_cooperative(
        self,
        lidars:            list,
        agent_poses_world: list,
        primary_idx:       int = 0,
        comm_range_m:      float = 70.0,
        max_cooperators:   int = 4,
    ) -> Detections:
        """Multi-agent cooperative forward pass.

        Parameters
        ----------
        lidars : list of (N_pts_i, 4) float32 ndarrays
            One point cloud per agent, each in its own ego LiDAR frame.
        agent_poses_world : list of [x, y, z, roll, yaw, pitch] (degrees)
            World pose of each agent's LiDAR sensor, same length and order
            as `lidars`.
        primary_idx : int
            Which agent in the input list is the primary ego. Output
            detections are returned in this agent's LiDAR frame.
        comm_range_m : float
            Drop cooperators whose LiDAR is farther than this from the
            primary's LiDAR (HEAL configs use 70 m).
        max_cooperators : int
            Cap how many cooperators (besides the primary) to fuse.

        Returns
        -------
        Detections in the primary agent's LiDAR frame.
        """
        if len(lidars) == 0 or len(lidars) != len(agent_poses_world):
            raise ValueError(
                f"len(lidars)={len(lidars)} must equal "
                f"len(agent_poses_world)={len(agent_poses_world)}"
            )
        if not (0 <= primary_idx < len(lidars)):
            raise ValueError(f"primary_idx {primary_idx} out of range for {len(lidars)} agents")

        # 1. Filter cooperators by distance from primary; reorder so primary is first.
        primary_pose = agent_poses_world[primary_idx]
        ordered = [(primary_idx, primary_pose)]
        for i, p in enumerate(agent_poses_world):
            if i == primary_idx:
                continue
            d = float(np.hypot(p[0] - primary_pose[0], p[1] - primary_pose[1]))
            if d <= comm_range_m:
                ordered.append((i, p))
        # Cap cooperator count (keep primary + up to max_cooperators others).
        ordered = ordered[: 1 + max_cooperators]

        N = len(ordered)
        # Cap by model's training-time max_cav.
        if N > self._max_cav:
            ordered = ordered[: self._max_cav]
            N = self._max_cav

        # 2. Voxelize each agent and stack with batch index.
        feat_chunks, coord_chunks, npts_chunks = [], [], []
        for batch_i, (orig_i, _pose) in enumerate(ordered):
            pcd = lidars[orig_i]
            if pcd.dtype != np.float32:
                pcd = pcd.astype(np.float32, copy=False)
            v = self.preprocessor.preprocess(pcd)
            coords = v["voxel_coords"]
            coords_b = np.concatenate(
                [np.full((coords.shape[0], 1), batch_i, dtype=coords.dtype), coords],
                axis=1,
            )
            feat_chunks.append(v["voxel_features"])
            coord_chunks.append(coords_b)
            npts_chunks.append(v["voxel_num_points"])

        voxel_features   = np.concatenate(feat_chunks,  axis=0)
        voxel_coords     = np.concatenate(coord_chunks, axis=0)
        voxel_num_points = np.concatenate(npts_chunks,  axis=0)

        inputs_m1 = {
            "voxel_features":   torch.from_numpy(voxel_features).float().to(self.device),
            "voxel_coords":     torch.from_numpy(voxel_coords).int().to(self.device),
            "voxel_num_points": torch.from_numpy(voxel_num_points).int().to(self.device),
        }

        # 3. Build pairwise transformation matrix in HEAL convention.
        # Entry [i, j] = transform from agent j's LiDAR frame to agent i's LiDAR frame.
        # Padded to (max_cav, max_cav, 4, 4); unused slots filled with identity.
        max_cav = self._max_cav
        pairwise = np.tile(np.eye(4, dtype=np.float32), (max_cav, max_cav, 1, 1))
        for i in range(N):
            for j in range(N):
                pose_i = ordered[i][1]
                pose_j = ordered[j][1]
                pairwise[i, j] = x1_to_x2(pose_j, pose_i).astype(np.float32)
        pairwise_t = torch.from_numpy(pairwise[None, ...]).to(self.device)  # (1, max_cav, max_cav, 4, 4)

        data_dict = {
            "agent_modality_list": ["m1"] * N,
            "record_len":          torch.tensor([N], dtype=torch.int).to(self.device),
            "pairwise_t_matrix":   pairwise_t,
            "inputs_m1":           inputs_m1,
        }

        # 4. Forward pass.
        out = self.model(data_dict)

        # 5. Decode (output is in primary agent's LiDAR frame because primary is batch 0).
        proc_in = {
            "ego": {
                "transformation_matrix": torch.eye(4).to(self.device),
                "anchor_box":            self._anchor_box,
            }
        }
        proc_out = {"ego": {k: v for k, v in out.items()
                            if k in ("cls_preds", "reg_preds", "dir_preds")}}
        pred_boxes, pred_scores = self.postprocessor.post_process(proc_in, proc_out)

        if pred_boxes is None or len(pred_boxes) == 0:
            return Detections(
                corners=np.zeros((0, 8, 3), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                centers_xy=np.zeros((0, 2), dtype=np.float32),
                P=0,
            )
        corners    = pred_boxes.cpu().numpy()
        scores     = pred_scores.cpu().numpy()
        centers_xy = corners.mean(axis=1)[:, :2]
        return Detections(corners=corners, scores=scores,
                          centers_xy=centers_xy, P=len(scores))
