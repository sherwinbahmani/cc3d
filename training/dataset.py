# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import math

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self, raw_idx):
        self._raw_labels = self._load_raw_labels(raw_idx) if self._use_labels else None
        self._raw_labels_std = None
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels(self._raw_idx[idx])
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels(self._raw_idx[idx]).copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels(0)
            self._label_shape = [sum([v.shape[0] for k,v in raw_labels.items()])]
            # if raw_labels.dtype == np.int64:
            #     self._label_shape = [int(np.max(raw_labels)) + 1]
            # else:
            #     self._label_shape = raw_labels.shape[1:]
        return self._label_shape

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        label_name = 'boxes.npz', # Name of image names
        max_coords = [1.0, 1.0, 1.0], # Max constraints for xyz lengths
        model_type = '3D', # TODO: Use from config same as in model
        load_voxel_masks = False, # Load pre-computed voxel grids
        images_sub_dir = 'images',
        labels_sub_dir = 'labels',
        num_samples_per_scene = 10, # TODO: Depending on boxes.npz
        remove_classes = None,
        optimize_camera_coords = False,
        img_pairs = False,
        flip_pair = False,
        use_traj = False,
        top_data = False,
        dataset_name = "3dfront",
        num_classes=None,
        add_floor_class=False,
        add_none_class=False,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self.max_coords = max_coords
        self.load_voxel_masks = load_voxel_masks
        self.model_type = model_type
        self.num_samples_per_scene = num_samples_per_scene
        self.remove_classes_idx = None
        self.img_pairs = img_pairs
        self.flip_pair = flip_pair
        self.use_traj = use_traj
        self.top_data = top_data
        self.dataset_name = dataset_name

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        self._label_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in ['.npz'])
        num_images = math.floor(len(self._image_fnames)/len(self._label_fnames))*(len(self._label_fnames))
        self._image_fnames = self._image_fnames[:num_images]
        self.intrinsics = None
        if self.top_data:
            self.num_samples_per_scene = 1
        else:
            with self._open_file(self._label_fnames[0]) as f:
                boxes = np.load(f)
                camera_coords_shape = boxes["camera_coords"].shape
                self.num_samples_per_scene = camera_coords_shape[0] * camera_coords_shape[1] // 3
                if self.dataset_name == "kitti":
                    self.intrinsics = boxes["intrinsic"]

        if len(self._label_fnames) == 0:
            raise IOError('No label files found in the specified path')
        if optimize_camera_coords:
            camera_coords = []
            for i in range(len(self._label_fnames)):
                fname = self._label_fnames[i]
                with self._open_file(fname) as f:
                    boxes = np.load(f)
                    camera_i, target_i = boxes["camera_coords"], boxes["target_coords"]
                    camera_coords.append(camera_i)
            self.camera_coords_params = torch.stack(camera_coords)
        else:
            self.camera_coords_params = None

        self.img_per_scene_ratio = len(self._image_fnames) / len(self._label_fnames)
        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        if num_classes is None:
            self.num_classes = self._load_raw_labels(0)['class_labels'].shape[-1]
            if add_floor_class:
                self.num_classes += 1
            if add_none_class:
                self.num_classes += 1
        else:
            self.num_classes = num_classes
        # if remove_classes is not None:
        #     class_dict = dict((v,k) for k,v in CLASSES_3DFRONT[self.num_classes].items())
        #     self.remove_classes_idx = [class_dict[class_name] for class_name in remove_classes]
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()
    
    @property
    def num_labels(self):
        return len(self._label_fnames)
    
    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels(0, 0)
            self._label_shape = [sum([v.shape[0] for k,v in raw_labels.items()])]
        return self._label_shape
    
    def _get_raw_labels(self, raw_idx, coords_idx=None, traj=False):
        self._raw_labels = self._load_raw_labels(raw_idx, coords_idx, traj=traj) if self._use_labels else None
        self._raw_labels_std = None
        return self._raw_labels

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)
    
    def get_label(self, idx, coords_idx, traj=False):
        label = self._get_raw_labels(self._raw_idx[idx], coords_idx, traj=traj)
        return label.copy()
    
    def get_image(self, idx, only_first=False):
        raw_idx = self._raw_idx[idx]
        image = self._load_raw_image(raw_idx, only_first)
        if self.img_pairs and only_first:
            img_ch_fac = 2
        else:
            img_ch_fac = 1
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == [self.image_shape[i]//ch_fac for i, ch_fac in enumerate([img_ch_fac, 1, 1])]
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy()
    
    def __getitem__(self, idx, only_first=False):
        raw_idx = self._raw_idx[idx]
        assert raw_idx == idx # TODO: Else check coords and image sampling with raw_idx
        image = self._load_raw_image(raw_idx)
        label_idx = math.floor(idx / self.img_per_scene_ratio)
        # Scale in case we have uneven number of labels per image in the dataset
        coords_idx = min(
            round((idx % self.img_per_scene_ratio) / self.img_per_scene_ratio * self.num_samples_per_scene),
            self.num_samples_per_scene - 1
            )
        if self.img_pairs and only_first:
            img_ch_fac = 2
        else:
            img_ch_fac = 1
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == [self.image_shape[i]//ch_fac for i, ch_fac in enumerate([img_ch_fac, 1, 1])]
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        image = image.copy()
        label = self.get_label(label_idx, coords_idx)
        if self.img_pairs and self.flip_pair and 0.5 > np.random.rand():
            image = np.concatenate((image[3:], image[:3]), 0)
            label['camera_coords'] = torch.cat((label['camera_coords'][3:], label['camera_coords'][:3]), 0)
        
        return image, label
        # return image.copy(), self.get_label(label_idx, coords_idx)
    
    def _get_raw_idx_pair(self, frame_idx, raw_idx):
        # TODO: Implement that only self.num_samples_per_scene - 1 can be sampled for coords 
        # Currently at last frame the previous frame is taken as there is no next frame
        if frame_idx == self.num_samples_per_scene - 1:
            raw_idx_pair = raw_idx - 1
        else:
            raw_idx_pair = raw_idx + 1
        return raw_idx_pair

    def _load_raw_image(self, raw_idx, only_first=False):
        fname = self._image_fnames[raw_idx]
        if self.img_pairs and not only_first:
            frame_idx = int(os.path.basename(fname).split('.')[0])
            raw_idx_pair = self._get_raw_idx_pair(frame_idx, raw_idx)
            fname_pair = self._image_fnames[raw_idx_pair]
            fnames = [fname, fname_pair]
        else:
            fnames = [fname]
        images = []
        for fname_i in fnames:
            with self._open_file(fname_i) as f:
                image = np.array(PIL.Image.open(f).convert('RGB'))
            if image.ndim == 2:
                image = image[:, :, np.newaxis] # HW => HWC
            image = image.transpose(2, 0, 1) # HWC => CHW
            images.append(image)
        images = np.concatenate(images, axis=0)
        return images
    
    def normalize_camera_coords(self, camera_coords, target_coords):
        max_coords = torch.tensor(self.max_coords, device=camera_coords.device)
        camera_coords = camera_coords/max_coords
        target_coords = target_coords/max_coords
        camera_coords[..., 1] -= 0.5
        target_coords[..., 1] -= 0.5
        camera_coords *= 2
        target_coords *= 2
        assert camera_coords.min() >= -1 and camera_coords.max() <= 1
        assert target_coords.min() >= -1 and target_coords.max() <= 1
        return camera_coords, target_coords

    def _load_raw_labels(self, raw_idx, coords_idx=None, traj=False, max_pad=20):
        fname = self._label_fnames[raw_idx]
        with self._open_file(fname) as f:
            boxes = np.load(f)
            boxes_keys = list(boxes.keys())
            if self.dataset_name == "3dfront" and "class_labels" in boxes_keys:
                class_labels=boxes["class_labels"]
                translations=boxes["translations"]
                sizes=boxes["sizes"]
                angles=boxes["angles"]
                room_layout = boxes["room_layout"]
            elif self.dataset_name == "kitti":
                semantic_layout = boxes["layout"]
            if (traj or (self.use_traj and 0.5 > np.random.rand())) and "camera_coords_traj" in boxes and "target_coords_traj" in boxes:
                camera_coords_all = boxes["camera_coords_traj"]
                target_coords_all = boxes["target_coords_traj"]
            else:
                camera_coords_all = boxes["camera_coords"]
                target_coords_all = boxes["target_coords"]
            if self.load_voxel_masks:
                voxel_masks = boxes["voxel_masks"]
        if coords_idx is None:
            coords_idx = np.random.randint(camera_coords_all.shape[0])
        camera_coords = camera_coords_all[min(coords_idx, camera_coords_all.shape[0]-1)]
        if self.img_pairs:
            coords_idx_pair = self._get_raw_idx_pair(coords_idx, coords_idx)
            camera_coords_pair = camera_coords_all[coords_idx_pair]
            camera_coords = np.concatenate((camera_coords, camera_coords_pair))
        target_coords = target_coords_all[min(coords_idx, camera_coords_all.shape[0]-1)]
        if self.remove_classes_idx is not None:
            class_indices = np.where(class_labels)[1]
            remove_classes_mask = np.isin(class_indices, self.remove_classes_idx) == False
            class_labels = class_labels[remove_classes_mask]
            translations = translations[remove_classes_mask]
            sizes = sizes[remove_classes_mask]
            angles = angles[remove_classes_mask]
        if self.dataset_name == "3dfront" and "class_labels" in boxes_keys:
            class_labels = torch.from_numpy(class_labels)
            translations = torch.from_numpy(translations)
            sizes = torch.from_numpy(sizes)
            angles = torch.from_numpy(angles)
            room_layout = torch.from_numpy(room_layout)
        elif self.dataset_name == "kitti":
            semantic_layout = torch.from_numpy(semantic_layout)
            camera_coords[-1] *=-1
            target_coords[-1] *=-1
        if self.dataset_name == "3dfront" and "class_labels" not in boxes_keys or self.dataset_name == "kitti":
            class_labels = torch.empty(0)
            translations = torch.empty(0)
            sizes = torch.empty(0)
            angles = torch.empty(0)
            room_layout = torch.empty(0)
        if self.dataset_name == "3dfront": # and "class_labels" not in boxes_keys:
            semantic_layout = torch.empty(0)
        camera_coords = torch.from_numpy(camera_coords)
        target_coords = torch.from_numpy(target_coords)
        if self.load_voxel_masks:
            voxel_masks = torch.from_numpy(voxel_masks)
        labels = {
            'class_labels': class_labels,
            'translations': translations,
            'sizes': sizes,
            'angles': angles,
            'room_layout': room_layout,
            'camera_coords': camera_coords.float(),
            'target_coords': target_coords.float(),
            'semantic_layout': semantic_layout,
            'label_idx': torch.tensor(raw_idx).unsqueeze(0),
            'coords_idx': torch.tensor(coords_idx).unsqueeze(0)
        }
        label_keys = ['class_labels', 'translations', 'sizes', 'angles']
        if self.load_voxel_masks:
            if self.model_type in ['2D', '2D_volume']:
                voxel_masks = voxel_masks[:, :,[-1],:]
            voxel_masks = torch.nn.functional.pad(voxel_masks, (0, 0, 0, 0, 0, 0, 0, max_pad-voxel_masks.shape[0]), 'constant', 0)
            labels['voxel_masks'] = voxel_masks
        
        if self.dataset_name == "3dfront" and "class_labels" in boxes_keys:
            for k, v in labels.items():
                if k in label_keys:
                    assert v.shape[0] <= max_pad, "Pad with higher number of objects, max pad reached"
                    labels[k] = torch.nn.functional.pad(v, (0, 0, 0, max_pad-v.shape[0]), 'constant', 0)
        return labels