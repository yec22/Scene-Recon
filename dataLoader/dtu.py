import torch, cv2, imageio
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from PIL import Image
from glob import glob
from torchvision import transforms as T
from .ray_utils import *

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

class DTUDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.img_wh = (int(1600/downsample),int(1200/downsample))
        self.downsample = downsample
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        
        self.read_meta()

        self.white_bg = True
        self.near_far = [2.5, 4.5]
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    
    def read_meta(self):
        render_cameras_name = 'cameras_sphere.npz'
        camera_dict = np.load(os.path.join(self.root_dir, render_cameras_name))
        self.camera_dict = camera_dict

        images_lis = sorted(glob(os.path.join(self.root_dir, 'image/*.png')))
        n_images = len(images_lis)
        masks_lis = sorted(glob(os.path.join(self.root_dir, 'mask/*.png')))
       
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

        if self.split == 'train':
            images_lis = images_lis[:-4]
            masks_lis = masks_lis[:-4]
            n_images -= 4
            self.world_mats_np = self.world_mats_np[:-4]
            self.scale_mats_np = self.scale_mats_np[:-4]
        else:
            images_lis = images_lis[-4:]
            masks_lis = masks_lis[-4:]
            n_images = 4
            self.world_mats_np = self.world_mats_np[-4:]
            self.scale_mats_np = self.scale_mats_np[-4:]

        intrinsics_all = []
        pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            intrinsics_all.append(torch.from_numpy(intrinsics).float())
            pose_all.append(torch.from_numpy(pose).float())

        w, h = self.img_wh
        self.focal = intrinsics_all[0][0, 0]
        self.focal /= self.downsample

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal, self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()

        self.poses = [] 
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []

        img_eval_interval = 1 if self.N_vis < 0 else n_images // self.N_vis
        idxs = list(range(0, n_images, img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):

            image_path = images_lis[i]
            mask_path = masks_lis[i]

            pose = pose_all[i]
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            img = Image.open(image_path) # rgb
            mask = Image.open(mask_path)
            
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
                mask = mask.resize(self.img_wh, Image.LANCZOS)
            
            img = self.transform(img)
            mask = self.transform(mask)
            img = img*mask + 1.0*(1-mask)

            img = img.permute(1, 2, 0) # (h, w, 3)
            img = img.reshape(h*w, 3)
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        self.poses = torch.stack(self.poses)

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        if self.split == 'train':
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}
        else:
            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            sample = {'rays': rays,
                      'rgbs': img}
        
        return sample