# Step 3: the whole study. Nothing to change here.
# It connects Google Drive, downloads the dataset (first run only), trains and tests
# every method, and saves tables, figures and a zip file in Drive.
# Settings come from the Settings cell; these lines only fill them in if that cell was skipped.
QUICK_TEST = globals().get('QUICK_TEST', False)
DRIVE_FOLDER = globals().get('DRIVE_FOLDER', '/content/drive/MyDrive/food_defects')
MVTEC2_FOLDER = globals().get('MVTEC2_FOLDER', None)
import argparse
import gc
import copy
import csv
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from collections import defaultdict


def _ensure(pkg, module):
    try:
        __import__(module)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)


_ensure("scikit-image", "skimage")

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from skimage.color import rgb2lab

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch

VERSION = "final"

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
SPLIT_URL = "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"
VISA_FOODS = ["cashew", "chewinggum", "fryum", "pipe_fryum", "macaroni1", "macaroni2"]
MVTEC2_FOODS = ["rice", "walnuts", "fruit_jelly"]
MVTEC_FOODS = ["hazelnut"]

DISPLAY = {
    "cashew": "Cashew", "chewinggum": "Chewing gum", "fryum": "Fryum",
    "pipe_fryum": "Pipe fryum", "macaroni1": "Macaroni 1", "macaroni2": "Macaroni 2",
    "rice": "Rice", "walnuts": "Walnuts", "fruit_jelly": "Fruit jelly", "hazelnut": "Hazelnut",
}

# Defect class ids in the VisA masks (from the dataset authors' id2class.py).
VISA_ID2CLASS = {
    "fryum": {1: "Burnt", 2: "Corner or Edge Breakage", 3: "Different Colour Spot",
              4: "Fryum Stuck Together", 5: "Middle Breakage", 6: "Similar Colour Spot",
              7: "Small Scratches", 8: "Other"},
    "cashew": {1: "Burnt", 2: "Corner or Edge Breakage", 3: "Different Colour Spot",
               4: "Middle Breakage", 5: "Small Holes", 6: "Small Scratches", 7: "Stuck Together",
               8: "Same Colour Spot", 9: "Other"},
    "chewinggum": {1: "Chunk of gum missing", 2: "Corner Missing", 3: "Scratches",
                   4: "Similar Colour Spot", 5: "Small Cracks", 6: "Other"},
    "macaroni1": {1: "Chip Around Edge And Corner", 2: "Different Colour Spot",
                  3: "Middle Breakage", 4: "Similar Colour Spot", 5: "Small Cracks",
                  6: "Small Scratches", 7: "Other"},
    "macaroni2": {1: "Breakage down the middle", 2: "Color spot similar to the Object",
                  3: "Different Color spot", 4: "Small chip around edge", 5: "Small Cracks",
                  6: "Small Scratches", 7: "Other"},
    "pipe_fryum": {1: "Burnt", 2: "Corner And Edge Breakage", 3: "Different Colour Spot",
                   4: "Middle Breakage", 5: "Similar Colour Spot", 6: "Small Scratches",
                   7: "Stuck Together", 8: "Small Cracks", 9: "Other"},
}

METHODS = ["fusion", "texture", "fusion_light", "colour", "padim", "ae"]
EXTRA_METHODS = ["fusion_max"]   # in the tables only: the v3 rule (colour always counts)
METHOD_LABEL = {
    "fusion": "CT-WRN50",
    "fusion_light": "CT-R18 (proposed)",
    "fusion_max": "CT-WRN50, fixed fusion",
    "texture_light": "PatchCore-R18",
    "texture": "PatchCore-WRN50",
    "colour": "CIELAB-ΔE",
    "padim": "PaDiM-R18",
    "ae": "CAE",
}
# Okabe-Ito colours (readable by colour-blind readers)
METHOD_COLOUR = {
    "fusion_light": "#C0392B", "fusion": "#E69F00", "texture": "#0072B2", "padim": "#009E73",
    "ae": "#8C8C8C", "colour": "#56B4E9", "texture_light": "#7FA7D1", "fusion_max": "#F5B041",
}
TEXTURE_RGB = np.array([213, 94, 0]) / 255.0   # alarm driven by texture or shape
COLOUR_RGB = np.array([0, 114, 178]) / 255.0   # alarm driven by colour

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------
# General helpers
# ----------------------------------------------------------------------------

def log(msg):
    print(time.strftime("[%H:%M:%S] ") + str(msg), flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def defect_group(name):
    """Merge similar defect names across foods into readable groups."""
    n = name.lower()
    if "burnt" in n:
        return "Burnt"
    if "stuck" in n:
        return "Stuck together"
    if ("similar" in n or "same" in n) and ("colo" in n or "spot" in n):
        return "Spot of similar colour"
    if "different" in n and "colo" in n:
        return "Spot of different colour"
    if "scratch" in n:
        return "Scratches"
    if "crack" in n:
        return "Cracks"
    if "hole" in n:
        return "Holes"
    if "missing" in n or "chunk" in n:
        return "Missing piece"
    if "break" in n or "chip" in n or "corner" in n or "edge" in n:
        return "Broken piece"
    if n in ("other", "defect", "bad"):
        return "Other"
    return name.replace("_", " ").strip().capitalize()


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

class Item:
    def __init__(self, path, label, mask_path=None, id2group=None, image_type=None):
        self.path = path
        self.label = int(label)            # 0 good, 1 defective
        self.mask_path = mask_path
        self.id2group = id2group or {}     # mask class id -> defect group
        self.image_type = image_type       # defect group for binary masks


def download(url, dest):
    if os.path.exists(dest):
        return dest
    tmp = dest + ".part"
    log(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done, step, nxt = 0, 1 << 20, 0.05
        while True:
            buf = r.read(step)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if total and done / total >= nxt:
                log(f"  {100 * done / total:.0f}% of {total / 1e9:.2f} GB")
                nxt += 0.05
    os.replace(tmp, dest)
    return dest


def find_visa_root(data_root):
    for cand in [data_root, os.path.join(data_root, "VisA")]:
        if os.path.isdir(os.path.join(cand, "cashew", "Data", "Images")):
            return cand
    for d in glob.glob(os.path.join(data_root, "*")) + glob.glob(os.path.join(data_root, "*", "*")):
        if os.path.isdir(os.path.join(d, "cashew", "Data", "Images")):
            return d
    return None


def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def connect_drive(folder):
    """Connect Google Drive in Colab and make the project folders."""
    if not os.path.isdir("/content/drive/MyDrive"):
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as e:
            sys.exit("Could not connect Google Drive (" + str(e) + ").\n"
                     "Run the notebook cells in Google Colab (Runtime > Run all).")
    for sub in ("", "data", "results"):
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
    log(f"Google Drive connected. Everything is kept in {folder}")


def prepare_visa(data_root, foods, skip_download, split_csv=None, keep_dir=None):
    """keep_dir holds the downloaded archive (Google Drive in Colab);
    data_root is where the food folders are unpacked (Colab's fast disk)."""
    keep_dir = keep_dir or data_root
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(keep_dir, exist_ok=True)
    root = find_visa_root(data_root)
    if root is None or any(not os.path.isdir(os.path.join(root, f)) for f in foods):
        if skip_download:
            sys.exit(f"VisA food folders not found under {data_root}. Remove --skip_download.")
        tar_path = os.path.join(keep_dir, "VisA_20220922.tar")
        if os.path.exists(tar_path):
            log(f"VisA archive already in {keep_dir}, no download needed")
        download(VISA_URL, tar_path)
        log("Unpacking the food folders from the VisA archive (a few minutes)")
        with tarfile.open(tar_path, "r:*") as tar:
            for m in tar:
                parts = [p for p in m.name.split("/") if p not in ("", ".")]
                if not parts or ".." in parts or m.name.startswith("/"):
                    continue
                if any(p in foods for p in parts[:2]) or parts[-1] == "split_csv":
                    try:
                        tar.extract(m, data_root, filter="data")
                    except TypeError:
                        tar.extract(m, data_root)
        root = find_visa_root(data_root)
        if root is None:
            sys.exit("Could not find the extracted VisA folders.")
    if split_csv is None:
        split_csv = os.path.join(keep_dir, "visa_1cls_split.csv")
        try:
            download(SPLIT_URL, split_csv)
        except Exception as e:
            log(f"Could not download the official split ({e}); using a fixed seeded split.")
            split_csv = None
    return root, split_csv


def load_visa(root, split_csv, food, seed):
    id2group = {k: defect_group(v) for k, v in VISA_ID2CLASS.get(food, {}).items()}
    train, test = [], []
    if split_csv and os.path.exists(split_csv):
        with open(split_csv) as f:
            for row in csv.DictReader(f):
                if row["object"] != food:
                    continue
                img = os.path.join(root, row["image"])
                if row["split"] == "train":
                    train.append(Item(img, 0))
                else:
                    lab = 0 if row["label"] == "normal" else 1
                    mp = os.path.join(root, row["mask"]) if row.get("mask") else None
                    test.append(Item(img, lab, mp, id2group))
    else:
        normal = sorted(glob.glob(os.path.join(root, food, "Data", "Images", "Normal", "*")))
        anomaly = sorted(glob.glob(os.path.join(root, food, "Data", "Images", "Anomaly", "*")))
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(normal))
        n_test = max(1, int(round(0.1 * len(normal))))
        for i in idx[n_test:]:
            train.append(Item(normal[i], 0))
        for i in idx[:n_test]:
            test.append(Item(normal[i], 0))
        for a in anomaly:
            stem = os.path.splitext(os.path.basename(a))[0]
            mp = os.path.join(root, food, "Data", "Masks", "Anomaly", stem + ".png")
            test.append(Item(a, 1, mp if os.path.exists(mp) else None, id2group))
    return train, None, test


def _imgs(folder):
    out = []
    for ext in ("png", "PNG", "jpg", "JPG", "jpeg", "bmp", "tif", "tiff"):
        out += glob.glob(os.path.join(folder, "*." + ext))
    return sorted(out)


def _find_mask(folder, stem):
    for name in (stem + "_mask.png", stem + ".png", stem + "_mask.PNG", stem + ".PNG"):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p
    return None


def load_mvtec2(root, food):
    base = os.path.join(root, food)
    train = [Item(p, 0) for p in _imgs(os.path.join(base, "train", "good"))]
    val = [Item(p, 0) for p in _imgs(os.path.join(base, "validation", "good"))] or None
    test = [Item(p, 0) for p in _imgs(os.path.join(base, "test_public", "good"))]
    gt = os.path.join(base, "test_public", "ground_truth", "bad")
    for p in _imgs(os.path.join(base, "test_public", "bad")):
        stem = os.path.splitext(os.path.basename(p))[0]
        test.append(Item(p, 1, _find_mask(gt, stem), image_type="Other"))
    return train, val, test


def load_mvtec(root, food):
    base = os.path.join(root, food)
    train = [Item(p, 0) for p in _imgs(os.path.join(base, "train", "good"))]
    test = []
    for d in sorted(glob.glob(os.path.join(base, "test", "*"))):
        kind = os.path.basename(d)
        for p in _imgs(d):
            if kind == "good":
                test.append(Item(p, 0))
            else:
                stem = os.path.splitext(os.path.basename(p))[0]
                mp = _find_mask(os.path.join(base, "ground_truth", kind), stem)
                test.append(Item(p, 1, mp, image_type=defect_group(kind)))
    return train, None, test


def load_rgb(path, size):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def load_mask(item, size):
    """Returns (class mask resized, list of defect groups present)."""
    if item.mask_path is None or not os.path.exists(item.mask_path):
        return None, ([item.image_type] if item.image_type else [])
    m = Image.open(item.mask_path)
    a = np.asarray(m)
    if a.ndim == 3:
        a = a[..., 0]
    ids = [int(v) for v in np.unique(a) if v != 0]
    if item.id2group and all(i in item.id2group for i in ids):
        groups = sorted({item.id2group[i] for i in ids})
        small = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((size, size), Image.NEAREST))
        return small.astype(np.uint8), groups
    small = np.asarray(Image.fromarray((a > 0).astype(np.uint8)).resize((size, size), Image.NEAREST))
    return small.astype(np.uint8), [item.image_type or "Other"]


class FoodData:
    """All images of one food, resized and held in memory."""

    def __init__(self, name, train, val, test, size, val_frac, seed, max_train=None):
        self.name = name
        rng = np.random.RandomState(seed)
        if val is None:
            idx = rng.permutation(len(train))
            n_val = max(10, int(round(val_frac * len(train))))
            n_val = min(n_val, max(1, len(train) // 3))
            val = [train[i] for i in idx[:n_val]]
            train = [train[i] for i in idx[n_val:]]
        if max_train and len(train) > max_train:
            idx = rng.permutation(len(train))[:max_train]
            train = [train[i] for i in sorted(idx)]
        self.train_items, self.val_items, self.test_items = train, val, test
        log(f"{DISPLAY.get(name, name)}: {len(train)} good for training, {len(val)} good for "
            f"setting the alarm level, {sum(t.label == 0 for t in test)} good and "
            f"{sum(t.label == 1 for t in test)} defective for testing")
        self.train = np.stack([load_rgb(t.path, size) for t in train])
        self.val = np.stack([load_rgb(t.path, size) for t in val])
        self.test = np.stack([load_rgb(t.path, size) for t in test])
        self.labels = np.array([t.label for t in test])
        self.masks, self.groups, self.has_mask = [], [], []
        for t in test:
            if t.label == 0:
                self.masks.append(np.zeros((size, size), np.uint8))
                self.groups.append([])
                self.has_mask.append(True)
            else:
                m, g = load_mask(t, size)
                self.has_mask.append(m is not None)
                self.masks.append(m if m is not None else np.zeros((size, size), np.uint8))
                self.groups.append(g)
        self.masks = np.stack(self.masks)
        self.has_mask = np.array(self.has_mask)
        self.id2group = {}
        for t in test:
            self.id2group.update(t.id2group)

    def group_mask(self, i, group):
        """Pixels of test image i that belong to a defect group."""
        m = self.masks[i]
        t = self.test_items[i]
        if t.id2group:
            ids = [k for k, v in t.id2group.items() if v == group]
            return np.isin(m, ids)
        return (m > 0) if group in self.groups[i] else np.zeros_like(m, bool)


# ----------------------------------------------------------------------------
# Networks and methods
# ----------------------------------------------------------------------------

def to_net_input(imgs, device):
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div(255.0)
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).to(device)


class Backbone(nn.Module):
    def __init__(self, arch, pretrained):
        super().__init__()
        if arch == "wide_resnet50_2":
            w = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
            m = torchvision.models.wide_resnet50_2(weights=w)
        else:
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            m = torchvision.models.resnet18(weights=w)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3 = m.layer1, m.layer2, m.layer3
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        f1 = self.layer1(self.stem(x))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return f1, f2, f3


def nearest_distance(q, bank, q_chunk=2048, b_chunk=65536):
    """Distance from each row of q to its nearest row in bank (both on one device)."""
    out = []
    for i in range(0, q.shape[0], q_chunk):
        qi = q[i:i + q_chunk]
        best = None
        for j in range(0, bank.shape[0], b_chunk):
            d = torch.cdist(qi, bank[j:j + b_chunk]).min(dim=1).values
            best = d if best is None else torch.minimum(best, d)
        out.append(best)
    return torch.cat(out)


def greedy_coreset(x, k, device, seed, proj_dim=128):
    """PatchCore greedy coreset: keeps k patches that cover the good-product variety.
    v6: no GPU-CPU sync inside the loop, so it runs several times faster."""
    n = x.shape[0]
    if k >= n:
        return torch.arange(n)
    g = torch.Generator().manual_seed(seed)
    proj = (torch.randn(x.shape[1], proj_dim, generator=g) / math.sqrt(proj_dim)).to(device)
    z = torch.cat([x[i:i + 65536].to(device).float() @ proj for i in range(0, n, 65536)])
    zn = (z * z).sum(1)
    start = int(torch.randint(n, (1,), generator=g).item())
    sel = torch.empty(k, dtype=torch.long, device=device)
    sel[0] = start
    mind = (zn - 2 * (z @ z[start]) + zn[start]).clamp_min(0)
    for t in range(1, k):
        j = torch.argmax(mind)
        sel[t] = j
        mind = torch.minimum(mind, (zn - 2 * (z @ z[j]) + zn[j]).clamp_min(0))
    return sel.cpu()


class TextureModel:
    """PatchCore-style: memory of good-product patch features from a pretrained network.
    v6: the memory is capped (bank_cap) so large images and big training sets stay fast."""

    def __init__(self, backbone, device, coreset, target_dim=1024, batch=16, seed=0,
                 bank_cap=40000, pool_cap=1000000, max_patches=None):
        self.bb, self.device, self.coreset = backbone, device, coreset
        self.dim, self.batch, self.seed = target_dim, batch, seed
        self.bank_cap, self.pool_cap, self.max_patches = bank_cap, pool_cap, max_patches
        self.bank = None

    @torch.no_grad()
    def embed(self, imgs, sample=False):
        """sample=True keeps at most max_patches random patches per image (training only).
        v6: fills one pre-allocated half-precision tensor, so memory is never doubled."""
        out = None
        pool = nn.AvgPool2d(3, 1, 1)
        g = torch.Generator().manual_seed(self.seed)
        for i in range(0, len(imgs), self.batch):
            x = to_net_input(imgs[i:i + self.batch], self.device)
            _, f2, f3 = self.bb(x)
            f2, f3 = pool(f2), pool(f3)
            f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            e = torch.cat([f2, f3], 1)
            b, c, h, w = e.shape
            dim = min(self.dim, c)
            e = e.permute(0, 2, 3, 1).reshape(-1, 1, c)
            if dim != c:
                e = F.adaptive_avg_pool1d(e, dim)
            e = e.reshape(b, h, w, dim)
            if sample and self.max_patches and h * w > self.max_patches:
                e = e.reshape(b, h * w, dim)
                keep = torch.stack([torch.randperm(h * w, generator=g)[: self.max_patches] for _ in range(b)])
                e = torch.gather(e, 1, keep.to(e.device).unsqueeze(-1).expand(-1, -1, dim))
            if out is None:
                out = torch.empty((len(imgs),) + tuple(e.shape[1:]), dtype=torch.float16)
            out[i:i + b] = e.half().cpu()
        return out

    @torch.no_grad()
    def score_imgs(self, imgs, chunk=32):
        """Patch alarm maps straight from images, a few at a time (features are not kept)."""
        return np.concatenate([self.score_from_emb(self.embed(imgs[i:i + chunk]))
                               for i in range(0, len(imgs), chunk)])

    def fit_from_emb(self, emb, seed=None):
        seed = self.seed if seed is None else seed
        x = emb.reshape(-1, emb.shape[-1])
        if x.shape[0] > self.pool_cap:
            g = torch.Generator().manual_seed(seed)
            x = x[torch.randperm(x.shape[0], generator=g)[: self.pool_cap]]
        k = int(min(max(1, round(x.shape[0] * self.coreset)), self.bank_cap))
        idx = greedy_coreset(x, k, self.device, seed)
        self.bank = x[idx].to(self.device).float()

    @torch.no_grad()
    def score_from_emb(self, emb):
        n, h, w, d = emb.shape
        maps = []
        for i in range(n):
            q = emb[i].reshape(-1, d).to(self.device).float()
            maps.append(nearest_distance(q, self.bank).reshape(h, w).cpu().numpy())
        return np.stack(maps)


def rgb_to_lab_t(x):
    """x: (B,3,H,W) float in [0,1], sRGB D65 -> CIELAB."""
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], device=x.device)
    t = torch.einsum("ij,bjhw->bihw", M, lin) / torch.tensor([0.95047, 1.0, 1.08883], device=x.device).view(1, 3, 1, 1)
    f = torch.where(t > 216 / 24389, t.clamp(min=1e-12) ** (1 / 3), (24389 / 27 * t + 16) / 116)
    return torch.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


class ColourModel:
    """Good-product colours in CIELAB. Each small patch is reduced to its mean L*, a*, b*.
    The alarm is the colour difference (CIE76 Delta E) to the nearest colour seen on good product.
    v6: the distance is pre-computed for every colour on a 1-unit grid (a look-up table),
    so scoring an image is a table look-up and takes well under a millisecond per patch."""

    L_N, AB_MIN, AB_N = 101, -128, 256

    def __init__(self, device, grid=32, min_count=1, seed=0, batch=32):
        self.device, self.grid, self.min_count, self.seed, self.batch = device, grid, min_count, seed, batch
        self.bank = None   # the Delta E look-up table

    @torch.no_grad()
    def embed(self, imgs):
        out = []
        s = imgs.shape[1]
        cell = s // self.grid
        for i in range(0, len(imgs), self.batch):
            x = torch.from_numpy(imgs[i:i + self.batch][:, : cell * self.grid, : cell * self.grid])
            x = x.to(self.device).permute(0, 3, 1, 2).float().div(255.0)
            lab = F.avg_pool2d(rgb_to_lab_t(x), cell)                 # B,3,grid,grid
            out.append(lab.permute(0, 2, 3, 1).cpu())
        return torch.cat(out)

    def _index(self, lab):
        L = lab[..., 0].round().clamp(0, self.L_N - 1).long()
        a = (lab[..., 1].round() - self.AB_MIN).clamp(0, self.AB_N - 1).long()
        b = (lab[..., 2].round() - self.AB_MIN).clamp(0, self.AB_N - 1).long()
        return (L * self.AB_N + a) * self.AB_N + b

    def fit_from_emb(self, emb, seed=None):
        idx = self._index(emb.reshape(-1, 3)).numpy()
        counts = np.bincount(idx, minlength=self.L_N * self.AB_N * self.AB_N)
        seen = (counts >= self.min_count).reshape(self.L_N, self.AB_N, self.AB_N)
        table = distance_transform_edt(~seen).astype(np.float32)   # Delta E to nearest good colour
        self.bank = torch.from_numpy(table.ravel()).to(self.device)

    @torch.no_grad()
    def score_from_emb(self, emb):
        idx = self._index(emb).to(self.bank.device)
        return self.bank[idx].cpu().numpy()


class PadimModel:
    """PaDiM: one Gaussian of good-product features at every image position."""

    def __init__(self, backbone, device, n_dims=100, batch=16, seed=0):
        self.bb, self.device, self.batch = backbone, device, batch
        g = torch.Generator().manual_seed(seed)
        self.idx = torch.randperm(64 + 128 + 256, generator=g)[:n_dims].to(device)

    @torch.no_grad()
    def _emb(self, imgs):
        x = to_net_input(imgs, self.device)
        f1, f2, f3 = self.bb(x)
        size = f1.shape[-2:]
        e = torch.cat([f1, F.interpolate(f2, size=size, mode="nearest"),
                       F.interpolate(f3, size=size, mode="nearest")], 1)
        e = e.index_select(1, self.idx)
        return e.reshape(e.shape[0], e.shape[1], -1), size

    @torch.no_grad()
    def fit(self, imgs):
        s1 = s2 = None
        n = 0
        for i in range(0, len(imgs), self.batch):
            e, self.size = self._emb(imgs[i:i + self.batch])
            e = e.double()
            s1 = e.sum(0) if s1 is None else s1 + e.sum(0)
            o = torch.einsum("bdp,bep->pde", e, e)
            if s2 is None:
                s2 = o
            else:
                s2.add_(o)        # in place: no third copy at 512 px
            del o
            n += e.shape[0]
        mean = s1 / n                                       # d, p
        mt = mean.t()
        for i in range(0, s2.shape[0], 2048):               # cov, in place and in chunks
            s2[i:i + 2048].sub_(n * mt[i:i + 2048, :, None] * mt[i:i + 2048, None, :])
        cov = s2.div_(max(n - 1, 1))
        eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
        self.mean = mean.float()
        # The inverse is done on the CPU: some Colab PyTorch builds have no GPU
        # kernel for torch.linalg.inv on the T4 ('no kernel image is available').
        # Done in chunks of positions so CPU memory stays small at 512 px.
        P = cov.shape[0]
        self.inv = torch.empty(cov.shape, dtype=torch.float32, device=cov.device)
        for i in range(0, P, 1024):
            self.inv[i:i + 1024] = torch.linalg.inv((cov[i:i + 1024] + 0.01 * eye).cpu()).float().to(cov.device)
        del s2, cov, s1, mt

    @torch.no_grad()
    def score(self, imgs):
        maps = []
        for i in range(0, len(imgs), self.batch):
            e, size = self._emb(imgs[i:i + self.batch])
            diff = e - self.mean.unsqueeze(0)
            m = torch.einsum("bdp,pde,bep->bp", diff, self.inv, diff).clamp_min(0).sqrt()
            maps.append(m.reshape(-1, *size).cpu().numpy())
        return np.concatenate(maps)


class AENet(nn.Module):
    def __init__(self):
        super().__init__()

        def down(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True))

        def up(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))

        self.enc = nn.Sequential(down(3, 32), down(32, 64), down(64, 128), down(128, 256),
                                 nn.Conv2d(256, 64, 3, 1, 1))
        self.dec = nn.Sequential(nn.Conv2d(64, 256, 3, 1, 1), nn.ReLU(True),
                                 up(256, 128), up(128, 64), up(64, 32),
                                 nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


class AEModel:
    """Convolutional autoencoder that learns to redraw good product."""

    def __init__(self, device, epochs=40, batch=16, lr=1e-3, seed=0):
        self.device, self.epochs, self.batch, self.lr, self.seed = device, epochs, batch, lr, seed

    def fit(self, imgs):
        torch.manual_seed(self.seed)
        self.net = AENet().to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.epochs))
        x_all = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div(255.0)
        n = len(x_all)
        self.net.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, self.batch):
                x = x_all[perm[i:i + self.batch]].to(self.device)
                loss = F.mse_loss(self.net(x), x)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(x)
            sched.step()
            if ep == 0 or (ep + 1) % 10 == 0 or ep == self.epochs - 1:
                log(f"    autoencoder epoch {ep + 1}/{self.epochs}, error {tot / n:.5f}")
        self.net.eval()

    @torch.no_grad()
    def score(self, imgs):
        maps = []
        for i in range(0, len(imgs), self.batch):
            x = torch.from_numpy(imgs[i:i + self.batch]).permute(0, 3, 1, 2).float().div(255.0).to(self.device)
            maps.append(((self.net(x) - x) ** 2).mean(1).cpu().numpy())
        return np.concatenate(maps)


def finish_maps(patch_maps, size, sigma):
    """Enlarge patch maps to image size and smooth them."""
    t = torch.from_numpy(np.ascontiguousarray(patch_maps)).float().unsqueeze(1)
    up = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[:, 0].numpy()
    if sigma > 0:
        up = np.stack([gaussian_filter(m, sigma=sigma) for m in up])
    return up.astype(np.float32)


# ----------------------------------------------------------------------------
# Scoring and metrics
# ----------------------------------------------------------------------------

class Result:
    """Maps and image scores of one method on one food."""

    def __init__(self, test_maps, val_scores, false_reject):
        self.scores = test_maps.reshape(len(test_maps), -1).max(1).astype(np.float64)
        self.maps = test_maps.astype(np.float16)   # half precision: maps are only drawn and pixel-scored
        self.val_scores = val_scores
        self.threshold = float(np.quantile(val_scores, 1.0 - false_reject))
        self.source = None  # for fusion: True where colour drove the alarm


def fuse(tex_test, tex_val, col_test, col_val, false_reject, margin=0.0):
    """Join texture and colour alarms after putting both on the scale of good product.
    The colour alarm counts only where it is more than `margin` standard deviations above
    the texture alarm: margin 0 is the v3 rule, margin inf is texture only."""
    tv = tex_val.reshape(len(tex_val), -1).max(1)
    cv = col_val.reshape(len(col_val), -1).max(1)
    mt, st = tv.mean(), tv.std() + 1e-8
    mc, sc = cv.mean(), cv.std() + 1e-8
    nt = (tex_test - mt) / st
    nc = (col_test - mc) / sc - margin
    fused = np.maximum(nt, nc)
    val_fused = np.maximum((tv - mt) / st, (cv - mc) / sc - margin)
    r = Result(fused.astype(np.float32), val_fused, false_reject)
    r.source = nc > nt
    r.margin = margin
    return r


MARGINS = [0.0, 0.5, 1.0, 2.0, 3.0, float("inf")]


def synthetic_defects(imgs, seed):
    """Fake defects painted on good images: a burnt patch, a spot of different colour,
    a scratch and a cut-and-paste piece. Used ONLY to choose the colour margin, so the
    choice never sees a real defect."""
    rng = np.random.RandomState(seed)
    out, kinds = [], []
    for img in imgs:
        s = img.shape[0]
        border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).astype(np.float32)
        bg = np.median(border, 0)
        fg = np.abs(img.astype(np.float32) - bg).sum(2) > 60
        ys, xs = np.nonzero(fg) if fg.mean() > 0.02 else np.nonzero(np.ones_like(fg))
        yy, xx = np.mgrid[0:s, 0:s]
        for kind in ("burnt", "colour spot", "scratch", "cut-paste"):
            im = img.astype(np.float32).copy()
            k = rng.randint(len(ys))
            cy, cx = ys[k], xs[k]
            if kind in ("burnt", "colour spot"):
                ry, rx = s * rng.uniform(0.02, 0.05, 2)
                d = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
                m = (np.clip(1.5 - d, 0, 1) * fg)[..., None]
                if kind == "burnt":
                    im = im * (1 - 0.65 * m) + 0.65 * m * np.array([60, 35, 15])
                else:
                    im = im * (1 - 0.7 * m) + 0.7 * m * (im * rng.uniform(0.55, 1.45, 3))
            elif kind == "scratch":
                L = s * rng.uniform(0.08, 0.2)
                ang = rng.uniform(0, np.pi)
                pil = Image.new("L", (s, s), 0)
                ImageDraw.Draw(pil).line([cx - L / 2 * np.cos(ang), cy - L / 2 * np.sin(ang),
                                          cx + L / 2 * np.cos(ang), cy + L / 2 * np.sin(ang)],
                                         fill=255, width=max(1, s // 200))
                m = (np.asarray(pil) / 255.0 * fg)[..., None]
                im = im * (1 - m) + m * im * (0.65 if rng.rand() < 0.5 else 1.3)
            else:
                p = max(4, int(s * rng.uniform(0.06, 0.12)))
                k2 = rng.randint(len(ys))
                sy, sx = np.clip([ys[k2] - p // 2, xs[k2] - p // 2], 0, s - p)
                ty, tx = np.clip([cy - p // 2, cx - p // 2], 0, s - p)
                im[ty:ty + p, tx:tx + p] = np.rot90(img[sy:sy + p, sx:sx + p].astype(np.float32), rng.randint(1, 4))
            out.append(np.clip(im, 0, 255).astype(np.uint8))
            kinds.append(kind)
    return np.stack(out), kinds


def choose_margin(tex_val, col_val, tex_syn, col_syn, false_reject):
    """Pick the colour margin that catches most fake defects at the plant alarm level
    (set on good images). Ties go to the larger margin (closer to texture only)."""
    best, rows = None, []
    for w in MARGINS:
        r = fuse(tex_syn, tex_val, col_syn, col_val, false_reject, w)
        caught = float(np.mean(r.scores >= r.threshold))
        rows.append({"margin": w, "fake_defects_caught": caught})
        if best is None or caught >= best[1] - 1e-9:
            best = (w, caught)
    return best[0], rows


def metrics(res, data, pixel_stride=None):
    lab = data.labels
    pixel_stride = pixel_stride or max(1, data.masks.shape[1] // 128)
    out = {}
    out["image_auroc"] = float(roc_auc_score(lab, res.scores)) if len(set(lab)) == 2 else float("nan")
    flagged = res.scores >= res.threshold
    out["defects_caught"] = float(flagged[lab == 1].mean()) if (lab == 1).any() else float("nan")
    out["good_rejected"] = float(flagged[lab == 0].mean()) if (lab == 0).any() else float("nan")
    keep = data.has_mask
    if keep.any() and (data.masks[keep] > 0).any():
        y = (data.masks[keep][:, ::pixel_stride, ::pixel_stride] > 0).ravel()
        s = res.maps[keep][:, ::pixel_stride, ::pixel_stride].ravel()
        out["pixel_auroc"] = float(roc_auc_score(y, s))
    else:
        out["pixel_auroc"] = float("nan")
    return out


# ----------------------------------------------------------------------------
# Running one food
# ----------------------------------------------------------------------------

def deletion_test(data, res, lt, col, lt_val, col_val, s, args, fr, wl, seed=0):
    """Explainable-AI faithfulness check (deletion test). For every caught defective item, blur away
    the flagged area and score the item again; as a control, blur an area of the same size and shape
    moved to a random place on the same item. A faithful explanation ends the alarm when its own area
    is removed, but not when a random area is removed."""
    rng = np.random.RandomState(seed)
    idx = [i for i in np.nonzero(data.labels == 1)[0] if res.scores[i] >= res.threshold]
    if not idx:
        return {"n_caught": 0}
    removed, control = [], []
    for i in idx:
        img = data.test[i]
        flag = res.maps[i].astype(np.float32) >= res.threshold
        flag = binary_dilation(flag, iterations=max(1, s // 64))
        blur = np.stack([gaussian_filter(img[..., c].astype(np.float32), sigma=s / 24) for c in range(3)], -1)
        a = img.copy(); a[flag] = blur[flag].astype(np.uint8)
        dy, dx = rng.randint(s // 4, 3 * s // 4, 2)
        moved = np.roll(np.roll(flag, dy, 0), dx, 1) & ~flag
        b = img.copy(); b[moved] = blur[moved].astype(np.uint8)
        removed.append(a); control.append(b)
    X = np.stack(removed + control)
    t = finish_maps(lt.score_imgs(X), s, args.sigma)
    c = finish_maps(col.score_from_emb(col.embed(X)), s, args.sigma)
    sc = fuse(t, lt_val, c, col_val, fr, wl).scores
    n = len(idx)
    before = res.scores[idx]
    after_flag, after_ctrl = sc[:n], sc[n:]
    return {"n_caught": n,
            "alarm_ends_flagged_removed_pct": round(100 * float(np.mean(after_flag < res.threshold)), 1),
            "alarm_ends_random_removed_pct": round(100 * float(np.mean(after_ctrl < res.threshold)), 1),
            "median_score_drop_flagged_pct": round(100 * float(np.median((before - after_flag) / np.maximum(before, 1e-6))), 1),
            "median_score_drop_random_pct": round(100 * float(np.median((before - after_ctrl) / np.maximum(before, 1e-6))), 1)}


def make_texture(bb, device, args, light=False):
    return TextureModel(bb, device, args.coreset, target_dim=384 if light else 1024, batch=args.batch,
                        seed=args.seed, bank_cap=args.light_bank_cap if light else args.bank_cap,
                        pool_cap=args.pool_cap, max_patches=args.max_patches)


def make_colour(device, args):
    return ColourModel(device, grid=max(4, args.img_size // 8), min_count=args.colour_min_count, seed=args.seed)


def run_food(data, args, device, tex_bb, padim_bb):
    s, fr = args.img_size, args.false_reject
    results, emb, info = {}, {}, {}
    syn, _ = synthetic_defects(data.val, args.seed)

    log(f"  texture model (PatchCore-style), images {s} x {s}")
    tex = make_texture(tex_bb, device, args)
    emb["tex_train"] = tex.embed(data.train, sample=True)
    t0 = time.time()
    tex.fit_from_emb(emb["tex_train"])
    log(f"    memory of {tex.bank.shape[0]} good patches ({time.time() - t0:.0f} s)")
    tex_val = finish_maps(tex.score_imgs(data.val), s, args.sigma)
    tex_test = finish_maps(tex.score_imgs(data.test), s, args.sigma)
    tex_syn = finish_maps(tex.score_imgs(syn), s, args.sigma)
    results["texture"] = Result(tex_test, tex_val.reshape(len(tex_val), -1).max(1), fr)

    log("  colour model (CIELAB, Delta E look-up table)")
    col = make_colour(device, args)
    emb["col_train"] = col.embed(data.train)
    emb["col_val"] = col.embed(data.val)
    emb["col_test"] = col.embed(data.test)
    col.fit_from_emb(emb["col_train"])
    col_val = finish_maps(col.score_from_emb(emb["col_val"]), s, args.sigma)
    col_test = finish_maps(col.score_from_emb(emb["col_test"]), s, args.sigma)
    col_syn = finish_maps(col.score_from_emb(col.embed(syn)), s, args.sigma)
    results["colour"] = Result(col_test, col_val.reshape(len(col_val), -1).max(1), fr)

    w, wrows = choose_margin(tex_val, col_val, tex_syn, col_syn, fr)
    info["margin"], info["margin_rows"] = w, wrows
    log("    fake-defect check: " + ", ".join(f"margin {r['margin']:g}: {100 * r['fake_defects_caught']:.0f}%" for r in wrows)
        + f"  -> colour margin {w:g}")
    results["fusion"] = fuse(tex_test, tex_val, col_test, col_val, fr, w)
    results["fusion_max"] = fuse(tex_test, tex_val, col_test, col_val, fr, 0.0)

    log("  light texture model (ResNet-18) for ordinary processors")
    lt = make_texture(padim_bb, device, args, light=True)
    lt.fit_from_emb(lt.embed(data.train, sample=True))
    lt_val = finish_maps(lt.score_imgs(data.val), s, args.sigma)
    lt_test = finish_maps(lt.score_imgs(data.test), s, args.sigma)
    lt_syn = finish_maps(lt.score_imgs(syn), s, args.sigma)
    wl, _ = choose_margin(lt_val, col_val, lt_syn, col_syn, fr)
    info["margin_light"] = wl
    results["fusion_light"] = fuse(lt_test, lt_val, col_test, col_val, fr, wl)
    info["xai"] = deletion_test(data, results["fusion_light"], lt, col, lt_val, col_val, s, args, fr, wl)
    del lt, lt_val, lt_test, lt_syn, tex_syn, col_syn, syn
    gc.collect()

    if not args.skip_padim:
        log("  PaDiM")
        pad = PadimModel(padim_bb, device, batch=args.batch, seed=args.seed)
        pad.fit(data.train)
        pv = finish_maps(pad.score(data.val), s, args.sigma)
        pt = finish_maps(pad.score(data.test), s, args.sigma)
        results["padim"] = Result(pt, pv.reshape(len(pv), -1).max(1), fr)
        del pad

    if not args.skip_ae:
        log("  autoencoder")
        ae = AEModel(device, epochs=args.ae_epochs, batch=args.batch, seed=args.seed)
        ae.fit(data.train)
        av = finish_maps(ae.score(data.val), s, args.sigma)
        at = finish_maps(ae.score(data.test), s, args.sigma)
        results["ae"] = Result(at, av.reshape(len(av), -1).max(1), fr)
        del ae

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results, emb, info


def defect_type_table(data, results):
    """Catch rate per defect group, and how often colour drove the alarm."""
    rows = []
    groups = sorted({g for gs in data.groups for g in gs})
    fus = results["fusion"]
    for g in groups:
        idx = [i for i, gs in enumerate(data.groups) if g in gs]
        if not idx:
            continue
        row = {"food": data.name, "group": g, "n_images": len(idx)}
        for m in ("fusion", "texture", "colour"):
            r = results[m]
            row[f"caught_{m}"] = float(np.mean(r.scores[idx] >= r.threshold))
        col_px = alarm_px = 0
        for i in idx:
            if not data.has_mask[i]:
                continue
            region = data.group_mask(i, g)
            alarm = region & (fus.maps[i] >= fus.threshold)
            alarm_px += int(alarm.sum())
            col_px += int((alarm & fus.source[i]).sum())
        row["alarm_pixels"] = alarm_px
        row["colour_pixels"] = col_px
        rows.append(row)
    return rows


def good_image_curve(data, emb, args, device, tex_bb, full_results, margin):
    """How many good images does a plant need? Re-fit on random subsets."""
    n_train = len(data.train)
    sizes = []
    for tok in args.sizes.split(","):
        tok = tok.strip()
        if tok == "all":
            sizes.append(n_train)
        elif tok and int(tok) < n_train:
            sizes.append(int(tok))
    sizes = sorted(set(sizes))
    rows = []
    s, fr = args.img_size, args.false_reject
    tex = make_texture(tex_bb, device, args)
    col = make_colour(device, args)
    for n in sizes:
        reps = 1 if n == n_train else args.repeats
        for r in range(reps):
            if n == n_train:   # the full set was already fitted in run_food
                for m in ("fusion", "texture"):
                    mm = metrics(full_results[m], data)
                    rows.append({"food": data.name, "method": m, "n_good": n, "repeat": r,
                                 "image_auroc": mm["image_auroc"], "defects_caught": mm["defects_caught"],
                                 "good_rejected": mm["good_rejected"]})
                log(f"    all {n} good images: detection score {100 * rows[-2]['image_auroc']:.1f}%")
                continue
            rng = np.random.RandomState(args.seed + 1000 * r + n)
            idx = np.sort(rng.choice(n_train, n, replace=False)) if n < n_train else np.arange(n_train)
            idx_t = torch.from_numpy(idx)
            tex.fit_from_emb(emb["tex_train"][idx_t], seed=args.seed + r)
            col.fit_from_emb(emb["col_train"][idx_t], seed=args.seed + r)
            tv = finish_maps(tex.score_imgs(data.val), s, args.sigma)
            tt = finish_maps(tex.score_imgs(data.test), s, args.sigma)
            cv = finish_maps(col.score_from_emb(emb["col_val"]), s, args.sigma)
            ct = finish_maps(col.score_from_emb(emb["col_test"]), s, args.sigma)
            res = {"fusion": fuse(tt, tv, ct, cv, fr, margin),
                   "texture": Result(tt, tv.reshape(len(tv), -1).max(1), fr)}
            for m, rr in res.items():
                mm = metrics(rr, data)
                rows.append({"food": data.name, "method": m, "n_good": n, "repeat": r,
                             "image_auroc": mm["image_auroc"], "defects_caught": mm["defects_caught"],
                             "good_rejected": mm["good_rejected"]})
            log(f"    {n} good images, repeat {r + 1}/{reps}: detection score "
                f"{100 * rows[-2]['image_auroc']:.1f}%")
    return rows


def time_methods(data, args, main_dev, backbones, n_images):
    """Time per image for each method, from photo to alarm map.
    Models are fitted once on main_dev, then timed on every device in backbones."""
    rows = []
    s = args.img_size
    imgs = data.test[:n_images]
    tex = make_texture(backbones[main_dev.type][0], main_dev, args)
    tex.fit_from_emb(tex.embed(data.train, sample=True))
    lt = make_texture(backbones[main_dev.type][1], main_dev, args, light=True)
    lt.fit_from_emb(lt.embed(data.train, sample=True))
    col = make_colour(main_dev, args)
    col.fit_from_emb(col.embed(data.train))
    pad = ae = None
    if not args.skip_padim:
        pad = PadimModel(backbones[main_dev.type][1], main_dev, batch=args.batch, seed=args.seed)
        pad.fit(data.train)
    if not args.skip_ae:
        ae = AEModel(main_dev, epochs=1, batch=args.batch, seed=args.seed)
        ae.fit(data.train)
    for dev_type, (tbb, pbb) in backbones.items():
        dev = torch.device(dev_type)
        tex.bb, tex.device, tex.bank, tex.batch = tbb, dev, tex.bank.to(dev), 1
        lt.bb, lt.device, lt.bank, lt.batch = pbb, dev, lt.bank.to(dev), 1
        col.device, col.bank = dev, col.bank.to(dev)
        fns = {
            "texture": lambda im: finish_maps(tex.score_from_emb(tex.embed(im)), s, args.sigma),
            "colour": lambda im: finish_maps(col.score_from_emb(col.embed(im)), s, args.sigma),
            "texture_light": lambda im: finish_maps(lt.score_from_emb(lt.embed(im)), s, args.sigma),
        }
        fns["fusion"] = lambda im: (fns["texture"](im), fns["colour"](im))
        fns["fusion_light"] = lambda im: (fns["texture_light"](im), fns["colour"](im))
        if pad is not None:
            pad.bb, pad.device, pad.batch = pbb, dev, 1
            pad.mean, pad.inv, pad.idx = pad.mean.to(dev), pad.inv.to(dev), pad.idx.to(dev)
            fns["padim"] = lambda im: finish_maps(pad.score(im), s, args.sigma)
        if ae is not None:
            ae.net.to(dev)
            ae.device, ae.batch = dev, 1
            fns["ae"] = lambda im: finish_maps(ae.score(im), s, args.sigma)
        for m, fn in fns.items():
            fn(imgs[:1])  # warm-up
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for i in range(len(imgs)):
                fn(imgs[i:i + 1])
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ms = 1000 * (time.time() - t0) / len(imgs)
            rows.append({"device": "GPU" if dev.type == "cuda" else "CPU", "method": m,
                         "ms_per_image": ms})
            log(f"    {m} on {'GPU' if dev.type == 'cuda' else 'CPU'}: {ms:.0f} ms per image")
    return rows


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def setup_fonts(font_path):
    name = None
    if font_path and os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        name = fm.FontProperties(fname=font_path).get_name()
    else:
        have = {f.name for f in fm.fontManager.ttflist}
        for cand in ("Times New Roman", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes",
                     "DejaVu Serif"):
            if cand in have:
                name = cand
                break
    if name not in ("Times New Roman", "Liberation Serif") and sys.platform.startswith("linux"):
        try:
            subprocess.run(["apt-get", "-qq", "install", "-y", "fonts-liberation"],
                           check=False, capture_output=True, timeout=300)
            p = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
            if os.path.exists(p):
                fm.fontManager.addfont(p)
                name = "Liberation Serif"
        except Exception:
            pass
    if name != "Times New Roman":
        log(f"Times New Roman not found, figures use {name} (same letter widths). To get Times "
            f"New Roman, put times.ttf from C:\\Windows\\Fonts in {DRIVE_FOLDER} and run again.")
    plt.rcParams.update({
        "font.family": name, "font.size": 8, "font.weight": "normal",
        "axes.titlesize": 8, "axes.labelsize": 8, "axes.titleweight": "normal",
        "axes.labelweight": "normal", "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "savefig.dpi": 600, "pdf.fonttype": 42,
    })


def save_fig(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}_{VERSION}.{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    log(f"  saved {name}_{VERSION}.png and .pdf")


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_examples(datas, all_results, out_dir):
    """Columns are foods; rows show the photo, the expert outline, the alarm map and its cause."""
    foods = [d for d in datas if (d.labels == 1).any()]
    n = len(foods)
    cell = min(7.2 / n, 1.6)
    w = cell * n
    fig, axes = plt.subplots(4, n, figsize=(w, cell * 4 + 0.45), squeeze=False)
    rows = ["Product", "Defect outlined\nby inspector", "Alarm map", "Why flagged"]
    for j, d in enumerate(foods):
        r = all_results[d.name]["fusion"]
        bad = np.where(d.labels == 1)[0]
        hit = [i for i in bad if r.scores[i] >= r.threshold] or list(bad)
        hit = sorted(hit, key=lambda i: r.scores[i])
        i = hit[len(hit) // 2]
        img = d.test[i] / 255.0
        gray = np.repeat(img.mean(2, keepdims=True), 3, axis=2)
        mask = d.masks[i] > 0
        alarm = r.maps[i] >= r.threshold
        heat = np.clip((r.maps[i] - r.threshold + 3) / 6.0, 0, 1)

        axes[0, j].imshow(img)
        axes[1, j].imshow(img)
        if mask.any():
            axes[1, j].contour(mask, levels=[0.5], colors=["#FFD700"], linewidths=0.8)
        axes[2, j].imshow(img)
        axes[2, j].imshow(heat, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
        why = gray * 0.75
        why[alarm & ~r.source[i]] = 0.25 * why[alarm & ~r.source[i]] + 0.75 * TEXTURE_RGB
        why[alarm & r.source[i]] = 0.25 * why[alarm & r.source[i]] + 0.75 * COLOUR_RGB
        axes[3, j].imshow(np.clip(why, 0, 1))
        axes[0, j].set_title(DISPLAY.get(d.name, d.name), pad=3)
        for k in range(4):
            axes[k, j].set_xticks([])
            axes[k, j].set_yticks([])
            for sp in axes[k, j].spines.values():
                sp.set_linewidth(0.4)
    for k in range(4):
        axes[k, 0].set_ylabel(rows[k], rotation=90, labelpad=4)
    fig.legend(handles=[Patch(color="#FFD700", label="Defect outlined by inspector"),
                        Patch(color=TEXTURE_RGB, label="Flagged for texture or shape"),
                        Patch(color=COLOUR_RGB, label="Flagged for colour")],
               loc="lower center", ncol=3 if n >= 4 else 1, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(wspace=0.04, hspace=0.04, bottom=0.07)
    save_fig(fig, out_dir, "fig1_examples")


def fig_methods(summary_rows, out_dir, methods, false_reject):
    foods = []
    for r in summary_rows:
        if r["food"] not in foods:
            foods.append(r["food"])
    get = {(r["food"], r["method"]): r for r in summary_rows}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    width = 0.8 / len(methods)
    x = np.arange(len(foods))
    panels = [("image_auroc", "AUROC (%)", "(a) Detection score over all alarm levels"),
              ("defects_caught", "Caught (%)",
               f"(b) Defective items caught when about {100 * false_reject:.0f}% of good items are rejected")]
    for ax, (key, ylabel, title) in zip(axes, panels):
        for k, m in enumerate(methods):
            vals = [100 * get[(f, m)][key] if (f, m) in get else np.nan for f in foods]
            ax.bar(x - 0.4 + width * (k + 0.5), vals, width * 0.92, color=METHOD_COLOUR[m],
                   label=METHOD_LABEL[m], edgecolor="none")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, linewidth=0.3, color="#CCCCCC")
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", pad=4)
        tidy(ax)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DISPLAY.get(f, f) for f in foods])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_fig(fig, out_dir, "fig2_methods_by_food")


def pooled_groups(type_rows):
    agg = defaultdict(lambda: {"n": 0, "fusion": 0.0, "texture": 0.0, "colour": 0.0, "apx": 0, "cpx": 0})
    for r in type_rows:
        a = agg[r["group"]]
        a["n"] += r["n_images"]
        for m in ("fusion", "texture", "colour"):
            a[m] += r[f"caught_{m}"] * r["n_images"]
        a["apx"] += r["alarm_pixels"]
        a["cpx"] += r["colour_pixels"]
    out = []
    for g, a in agg.items():
        out.append({"group": g, "n_images": a["n"],
                    "caught_fusion": a["fusion"] / a["n"], "caught_texture": a["texture"] / a["n"],
                    "caught_colour": a["colour"] / a["n"],
                    "colour_share": a["cpx"] / a["apx"] if a["apx"] else np.nan})
    return sorted(out, key=lambda r: r["caught_fusion"])


def fig_defect_types(pooled, out_dir):
    pooled = [p for p in pooled if p["n_images"] >= 3]
    if not pooled:
        return
    labels = [f"{p['group']} (n = {p['n_images']})" for p in pooled]
    y = np.arange(len(pooled))
    h = max(2.0, 0.24 * len(pooled) + 0.8)

    fig, ax = plt.subplots(figsize=(3.5, h))
    ax.barh(y + 0.2, [100 * p["caught_texture"] for p in pooled], 0.38,
            color=METHOD_COLOUR["texture"], label=METHOD_LABEL["texture"])
    ax.barh(y - 0.2, [100 * p["caught_fusion"] for p in pooled], 0.38,
            color=METHOD_COLOUR["fusion"], label=METHOD_LABEL["fusion"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Defective items caught (%)")
    ax.xaxis.grid(True, linewidth=0.3, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=1)
    tidy(ax)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig3_defect_types")

    keep = [p for p in pooled if not np.isnan(p["colour_share"])]
    if not keep:
        return
    keep = sorted(keep, key=lambda p: p["colour_share"])
    y = np.arange(len(keep))
    fig, ax = plt.subplots(figsize=(3.5, max(2.0, 0.24 * len(keep) + 0.8)))
    c = [100 * p["colour_share"] for p in keep]
    ax.barh(y, c, 0.6, color=COLOUR_RGB, label="Colour")
    ax.barh(y, [100 - v for v in c], 0.6, left=c, color=TEXTURE_RGB, label="Texture or shape")
    ax.set_yticks(y)
    ax.set_yticklabels([p["group"] for p in keep])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of alarm area inside the defect (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2)
    tidy(ax)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig4_why_flagged")


def fig_good_images(curve_rows, out_dir):
    if not curve_rows:
        return
    rows = [r for r in curve_rows if r["method"] == "fusion"]
    foods = []
    for r in rows:
        if r["food"] not in foods:
            foods.append(r["food"])
    palette = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#999999",
               "#000000", "#F0E442", "#8C564B"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, key, ylab, tag in ((axes[0], "image_auroc", "Detection score, AUROC (%)", "(a)"),
                               (axes[1], "defects_caught", "Defective items caught (%)", "(b)")):
        full = {f: max(r["n_good"] for r in rows if r["food"] == f) for f in foods}
        part = sorted({r["n_good"] for r in rows if r["n_good"] != full[r["food"]]})
        x_all = 2 * max(part) if part else 1
        for k, f in enumerate(foods):
            fr = [r for r in rows if r["food"] == f]
            ns = sorted({r["n_good"] for r in fr})
            xs = [x_all if n == full[f] else n for n in ns]
            mean = np.array([100 * np.mean([r[key] for r in fr if r["n_good"] == n]) for n in ns])
            sd = np.array([100 * np.std([r[key] for r in fr if r["n_good"] == n]) for n in ns])
            col = palette[k % len(palette)]
            ax.plot(xs, mean, "-o", color=col, markersize=2.5, linewidth=1.0,
                    label=f"{DISPLAY.get(f, f)} (all = {full[f]})")
            ax.fill_between(xs, mean - sd, mean + sd, color=col, alpha=0.15, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks(part + [x_all])
        ax.set_xticklabels([str(t) for t in part] + ["All"])
        ax.minorticks_off()
        ax.set_xlabel("Good images used for training")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, linewidth=0.3, color="#CCCCCC")
        ax.set_axisbelow(True)
        ax.text(-0.16, 1.02, tag, transform=ax.transAxes, va="bottom")
        tidy(ax)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    save_fig(fig, out_dir, "fig5_good_images_needed")


def fig_speed(speed_rows, out_dir, methods):
    if not speed_rows:
        return
    devs = [d for d in ("GPU", "CPU") if any(r["device"] == d for r in speed_rows)]
    ms = [m for m in methods + ["texture_light"] if any(r["method"] == m for r in speed_rows)]
    get = {(r["device"], r["method"]): r["ms_per_image"] for r in speed_rows}
    fig, ax = plt.subplots(figsize=(3.5, 0.3 * len(ms) * len(devs) + 0.9))
    y = np.arange(len(ms))
    hgt = 0.8 / len(devs)
    dev_col = {"GPU": "#0072B2", "CPU": "#E69F00"}
    vals_all = []
    for k, d in enumerate(devs):
        vals = [get.get((d, m), np.nan) for m in ms]
        vals_all += [v for v in vals if not np.isnan(v)]
        yy = y + 0.4 - hgt * (k + 0.5)
        ax.barh(yy, vals, hgt * 0.9, color=dev_col[d],
                label=("Graphics card (GPU)" if d == "GPU" else "Ordinary processor (CPU)"))
        for yi, v in zip(yy, vals):
            if not np.isnan(v):
                ax.text(v * 1.08, yi, f"{v:.0f} ms" if v >= 1 else f"{v:.1f} ms", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABEL[m] for m in ms])
    ax.set_xscale("log")
    lo, hi = min(vals_all), max(vals_all)
    ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
             if lo / 2 <= t <= hi * 3]
    ax.set_xlim(lo / 2, hi * 4)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Time per image (ms, log scale)")
    ax.invert_yaxis()
    ax.legend(loc="lower center", bbox_to_anchor=(0.3, 1.0), ncol=1)
    tidy(ax)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig6_speed")


def fig_roc_pr(score_rows, out_dir, methods):
    """Fig. 9 ROC curves, Fig. 10 precision-recall curves (one panel per food, one line per method),
    Fig. 11 score distributions of the light version with its alarm level."""
    foods = []
    for r in score_rows:
        if r["food"] not in foods:
            foods.append(r["food"])
    get = defaultdict(list)
    for r in score_rows:
        get[(r["food"], r["method"])].append(r)
    nc = 3
    nr = int(math.ceil(len(foods) / nc))
    out = []
    for kind in ("roc", "pr"):
        fig, axes = plt.subplots(nr, nc, figsize=(7.2, 2.35 * nr + 0.5), squeeze=False)
        for k, f in enumerate(foods):
            ax = axes[k // nc, k % nc]
            for m in methods:
                rr = get.get((f, m))
                if not rr:
                    continue
                y = np.array([r["label"] for r in rr]); s = np.array([r["score"] for r in rr]); thr = rr[0]["threshold"]
                lw = 1.6 if m == "fusion_light" else 0.9
                if kind == "roc":
                    fpr, tpr, _ = roc_curve(y, s)
                    auc = roc_auc_score(y, s)
                    ax.plot(fpr, tpr, color=METHOD_COLOUR[m], lw=lw, label=f"{METHOD_LABEL[m]} {100 * auc:.1f}")
                    op_x, op_y = np.mean(s[y == 0] >= thr), np.mean(s[y == 1] >= thr)
                    if m == "fusion_light":
                        ax.plot(op_x, op_y, "o", ms=4, color=METHOD_COLOUR[m], mec="black", mew=0.5, zorder=5)
                    out.append({"food": f, "method": m, "auroc": auc, "average_precision": average_precision_score(y, s),
                                "fpr_at_alarm": op_x, "tpr_at_alarm": op_y})
                else:
                    pr, rc, _ = precision_recall_curve(y, s)
                    ap_ = average_precision_score(y, s)
                    ax.plot(rc, pr, color=METHOD_COLOUR[m], lw=lw, label=f"{METHOD_LABEL[m]} {100 * ap_:.1f}")
            if kind == "roc":
                ax.plot([0, 1], [0, 1], ls=":", lw=0.6, color="#999999")
                ax.set_xlabel("False rejection rate", fontsize=7)
                ax.set_ylabel("Recall (true positive rate)", fontsize=7)
            else:
                base = np.mean(np.array([r["label"] for r in get[(f, methods[0])]]))
                ax.axhline(base, ls=":", lw=0.6, color="#999999")
                ax.set_xlabel("Recall", fontsize=7)
                ax.set_ylabel("Precision", fontsize=7)
                ax.set_ylim(0.4, 1.01)
            ax.set_xlim(-0.01, 1.01)
            ax.set_title(DISPLAY.get(f, f), fontsize=8, pad=3)
            ax.legend(fontsize=5.6, loc="lower right" if kind == "roc" else "lower left", handlelength=1.4,
                      title=("AUROC (%)" if kind == "roc" else "Average precision (%)"), title_fontsize=5.8)
            ax.tick_params(labelsize=6.5)
            tidy(ax)
        for k in range(len(foods), nr * nc):
            axes[k // nc, k % nc].axis("off")
        fig.tight_layout()
        save_fig(fig, out_dir, "fig9_roc" if kind == "roc" else "fig10_precision_recall")
    # Fig. 11: score distributions, light version
    fig, axes = plt.subplots(nr, nc, figsize=(7.2, 2.1 * nr + 0.4), squeeze=False)
    for k, f in enumerate(foods):
        ax = axes[k // nc, k % nc]
        rr = get.get((f, "fusion_light"), [])
        if not rr:
            ax.axis("off"); continue
        y = np.array([r["label"] for r in rr]); s = np.array([r["score"] for r in rr]); thr = rr[0]["threshold"]
        lo, hi = np.percentile(s, 0.5), np.percentile(s, 99.5)
        bins = np.linspace(lo, hi, 30)
        ax.hist(np.clip(s[y == 0], lo, hi), bins=bins, color="#9AA5A0", alpha=0.85, label="Normal")
        ax.hist(np.clip(s[y == 1], lo, hi), bins=bins, color=METHOD_COLOUR["fusion_light"], alpha=0.75, label="Defective")
        ax.axvline(thr, color="black", lw=0.9, ls="--", label="Decision threshold")
        ax.set_title(DISPLAY.get(f, f), fontsize=8, pad=3)
        ax.set_xlabel("Anomaly score z (SD above normal)", fontsize=7)
        ax.set_ylabel("Number of images", fontsize=7)
        ax.tick_params(labelsize=6.5)
        if k == 0:
            ax.legend(fontsize=6)
        tidy(ax)
    for k in range(len(foods), nr * nc):
        axes[k // nc, k % nc].axis("off")
    fig.tight_layout()
    save_fig(fig, out_dir, "fig11_score_distributions")
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    log(f"  saved {os.path.basename(path)}")


def main():
    ap = argparse.ArgumentParser(description="Good-product-only camera inspection of food defects")
    ap.add_argument("--data_root", default=None, help="where the food folders are unpacked")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--no_drive", action="store_true", help="do not connect Google Drive in Colab")
    ap.add_argument("--foods", default=",".join(VISA_FOODS),
                    help="VisA foods to use, comma separated")
    ap.add_argument("--mvtec2_dir", default=None, help="folder holding MVTec AD 2 rice, walnuts, fruit_jelly")
    ap.add_argument("--mvtec_dir", default=None, help="folder holding MVTec AD hazelnut")
    ap.add_argument("--split_csv", default=None, help="VisA 1cls split file, if already downloaded")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--coreset", type=float, default=0.1, help="share of good patches kept in memory")
    ap.add_argument("--bank_cap", type=int, default=40000, help="most patches kept in the texture memory")
    ap.add_argument("--light_bank_cap", type=int, default=10000, help="memory size of the light model")
    ap.add_argument("--pool_cap", type=int, default=1000000, help="most training patches considered")
    ap.add_argument("--max_patches", type=int, default=1024, help="training patches kept per image")
    ap.add_argument("--hires_foods", default="macaroni1,macaroni2",
                    help="foods photographed as several pieces per image, run at --hires_size")
    ap.add_argument("--hires_size", type=int, default=512)
    ap.add_argument("--colour_min_count", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=4.0, help="smoothing of alarm maps, in pixels")
    ap.add_argument("--false_reject", type=float, default=0.05,
                    help="share of good items allowed to be rejected when setting the alarm level")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--ae_epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sizes", default="10,25,50,100,200,all")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--speed_images", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--font_path", default=None)
    ap.add_argument("--max_train", type=int, default=None, help="cap on good training images per food")
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--skip_padim", action="store_true")
    ap.add_argument("--skip_ae", action="store_true")
    ap.add_argument("--skip_curve", action="store_true")
    ap.add_argument("--skip_speed", action="store_true")
    ap.add_argument("--quick", action="store_true", help="small fast run to check everything works")
    ap.add_argument("--no_pretrained", action="store_true", help="testing only: random network weights")
    args, _ = ap.parse_known_args()   # ignores the extra arguments Jupyter adds
    # v9 = the v6 study, run again only to save every test image's score for ROC / PR curves
    args.skip_curve, args.skip_speed = True, True

    if QUICK_TEST:
        args.quick = True
    colab = in_colab()
    keep_dir = None
    if colab and not args.no_drive:
        connect_drive(DRIVE_FOLDER)
        keep_dir = os.path.join(DRIVE_FOLDER, "data")
        if args.out_dir is None:
            args.out_dir = os.path.join(DRIVE_FOLDER, "results",
                                        ("quick_" if args.quick else "full_") + VERSION)
        if args.font_path is None:
            for p in (os.path.join(DRIVE_FOLDER, "times.ttf"), "/content/times.ttf"):
                if os.path.exists(p):
                    args.font_path = p
                    break
        if args.mvtec2_dir is None and MVTEC2_FOLDER:
            args.mvtec2_dir = MVTEC2_FOLDER
    if args.data_root is None:
        args.data_root = "/content/data" if colab else "data_goodonly"
    if args.out_dir is None:
        args.out_dir = f"results_goodonly_{'quick_' if args.quick else ''}{VERSION}"

    if args.quick:
        args.coreset = min(args.coreset, 0.01)
        args.ae_epochs = min(args.ae_epochs, 5)
        args.repeats = 1
        args.sizes = "10,50,all"
        args.speed_images = 3
        if args.foods == ",".join(VISA_FOODS):
            args.foods = "cashew,fryum"

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    setup_fonts(args.font_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"goodonly_camera_{VERSION}: running on {'GPU' if device.type == 'cuda' else 'CPU'}")
    if device.type != "cuda" and colab:
        log("No GPU found. In Colab choose Runtime > Change runtime type > T4 GPU and run again. "
            "Continuing on CPU, which is very slow.")
    log(f"Results will be saved in {args.out_dir}")
    with open(os.path.join(args.out_dir, f"settings_{VERSION}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    tex_bb = Backbone("wide_resnet50_2", not args.no_pretrained).to(device)
    padim_bb = Backbone("resnet18", not args.no_pretrained).to(device)

    # ---- data
    specs = []
    visa_foods = [f.strip() for f in args.foods.split(",") if f.strip()]
    if visa_foods:
        root, split_csv = prepare_visa(args.data_root, visa_foods, args.skip_download, args.split_csv,
                                       keep_dir=keep_dir)
        for f in visa_foods:
            specs.append((f,) + load_visa(root, split_csv, f, args.seed))
    if args.mvtec2_dir:
        for f in MVTEC2_FOODS:
            if os.path.isdir(os.path.join(args.mvtec2_dir, f)):
                specs.append((f,) + load_mvtec2(args.mvtec2_dir, f))
            else:
                log(f"MVTec AD 2 {f} not found in {args.mvtec2_dir}, skipped")
    if args.mvtec_dir:
        for f in MVTEC_FOODS:
            if os.path.isdir(os.path.join(args.mvtec_dir, f)):
                specs.append((f,) + load_mvtec(args.mvtec_dir, f))

    methods = [m for m in METHODS if not (m == "padim" and args.skip_padim) and not (m == "ae" and args.skip_ae)]
    summary, type_rows, curve_rows, datas, all_results, margins = [], [], [], [], {}, {}
    score_rows = []
    xai_rows = []

    for name, train, val, test in specs:
        if not train or not test:
            log(f"{name}: no images found, skipped")
            continue
        log(f"=== {DISPLAY.get(name, name)} ===")
        fargs = copy.copy(args)
        if name in [f.strip() for f in args.hires_foods.split(",")]:
            fargs.img_size = args.hires_size
        fargs.sigma = args.sigma * fargs.img_size / 256.0
        data = FoodData(name, train, val, test, fargs.img_size, args.val_frac, args.seed, args.max_train)
        results, emb, info = run_food(data, fargs, device, tex_bb, padim_bb)
        margins[name] = (fargs.img_size, info["margin"], info["margin_light"], info["margin_rows"])
        xai_rows.append({"food": name, **info["xai"]})
        log(f"  XAI deletion test (light): {info['xai']}")
        write_csv(os.path.join(args.out_dir, f"xai_deletion_test_{VERSION}.csv"), xai_rows)
        for m in methods + EXTRA_METHODS:
            mm = metrics(results[m], data)
            mm.update({"food": name, "method": m, "threshold": results[m].threshold,
                       "img_size": fargs.img_size,
                       "colour_margin": getattr(results[m], "margin", "")})
            summary.append(mm)
            log(f"  {METHOD_LABEL[m]:38s} detection score {100 * mm['image_auroc']:5.1f}%  "
                f"pixel score {100 * mm['pixel_auroc']:5.1f}%  caught {100 * mm['defects_caught']:5.1f}%  "
                f"good rejected {100 * mm['good_rejected']:4.1f}%")
        for m in methods + EXTRA_METHODS:
            for k, it in enumerate(data.test_items):
                score_rows.append({"food": name, "method": m, "image": os.path.relpath(it.path, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(it.path))))),
                                   "label": int(data.labels[k]), "score": float(results[m].scores[k]),
                                   "threshold": float(results[m].threshold)})
        write_csv(os.path.join(args.out_dir, f"per_image_scores_{VERSION}.csv"), score_rows)
        type_rows += defect_type_table(data, results)
        if not args.skip_curve:
            log("  how many good images are needed")
            curve_rows += good_image_curve(data, emb, fargs, device, tex_bb, results, info["margin"])
        # keep only what the example figure needs
        for m in list(results):
            if m != "fusion":
                results[m].maps = None
        results["fusion"].maps = results["fusion"].maps.astype(np.float16)
        all_results[name] = results
        datas.append(data)
        data.train = data.val = None
        del emb, results
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        write_csv(os.path.join(args.out_dir, f"results_by_food_{VERSION}.csv"), summary)

    if not summary:
        sys.exit("No food was processed.")

    # ---- speed on one food
    speed_rows = []
    if not args.skip_speed:
        log("Timing each method")
        name, train, val, test = specs[0]
        d0 = FoodData(name, train, val, test, args.img_size, args.val_frac, args.seed, args.max_train)
        backbones = {device.type: (tex_bb, padim_bb)}
        if device.type == "cuda":
            backbones["cpu"] = (Backbone("wide_resnet50_2", not args.no_pretrained),
                                Backbone("resnet18", not args.no_pretrained))
        speed_rows = time_methods(d0, args, device, backbones, args.speed_images)

    # ---- averages over foods
    mean_rows = []
    for m in methods + EXTRA_METHODS:
        rs = [r for r in summary if r["method"] == m]
        mean_rows.append({"method": m, "label": METHOD_LABEL[m],
                          **{k: float(np.nanmean([r[k] for r in rs]))
                             for k in ("image_auroc", "pixel_auroc", "defects_caught", "good_rejected")}})
    pooled = pooled_groups(type_rows)

    log("Saving tables")
    write_csv(os.path.join(args.out_dir, f"results_by_food_{VERSION}.csv"), summary)
    write_csv(os.path.join(args.out_dir, f"results_mean_{VERSION}.csv"), mean_rows)
    write_csv(os.path.join(args.out_dir, f"defect_types_by_food_{VERSION}.csv"), type_rows)
    write_csv(os.path.join(args.out_dir, f"defect_types_pooled_{VERSION}.csv"), pooled)
    write_csv(os.path.join(args.out_dir, f"good_images_needed_{VERSION}.csv"), curve_rows)
    write_csv(os.path.join(args.out_dir, f"speed_{VERSION}.csv"), speed_rows)

    write_csv(os.path.join(args.out_dir, f"per_image_scores_{VERSION}.csv"), score_rows)
    log("Drawing ROC, precision-recall and score-distribution figures")
    roc_rows = fig_roc_pr(score_rows, args.out_dir, methods)
    write_csv(os.path.join(args.out_dir, f"roc_auc_ap_{VERSION}.csv"), roc_rows)
    log("Drawing figures")
    fig_examples(datas, all_results, args.out_dir)
    fig_methods(summary, args.out_dir, methods, args.false_reject)
    fig_defect_types(pooled, args.out_dir)
    fig_good_images(curve_rows, args.out_dir)
    fig_speed(speed_rows, args.out_dir, methods)

    lines = [f"goodonly_camera_{VERSION} summary", "",
             "Average over foods (detection score = image AUROC; caught = defective items flagged "
             f"when the alarm level is set so that about {100 * args.false_reject:.0f}% of good items are rejected)", ""]
    for r in mean_rows:
        lines.append(f"{r['label']:38s} detection {100 * r['image_auroc']:5.1f}%  pixel {100 * r['pixel_auroc']:5.1f}%  "
                     f"caught {100 * r['defects_caught']:5.1f}%  good rejected {100 * r['good_rejected']:4.1f}%")
    lines += ["", "Per food: image size, colour margin chosen on fake defects (inf = texture only)"]
    for f, (sz, w, wl, wrows) in margins.items():
        lines.append(f"  {DISPLAY.get(f, f):12s} {sz} px   margin {w:g} (light {wl:g})   fake defects caught: "
                     + ", ".join(f"{r['margin']:g}->{100 * r['fake_defects_caught']:.0f}%" for r in wrows))
    lines += ["", "Explainable AI - deletion test (light version, caught defects): alarm ends when the FLAGGED area "
              "is blurred away vs when a RANDOM area of the same size is blurred away"]
    for r in xai_rows:
        if r.get("n_caught"):
            lines.append(f"  {DISPLAY.get(r['food'], r['food']):12s} n={r['n_caught']:3d}  flagged removed: alarm ends "
                         f"{r['alarm_ends_flagged_removed_pct']:5.1f}% (median drop {r['median_score_drop_flagged_pct']:.0f}%)   "
                         f"random removed: {r['alarm_ends_random_removed_pct']:5.1f}% (median drop {r['median_score_drop_random_pct']:.0f}%)")
    tot = sum(r.get("n_caught", 0) for r in xai_rows)
    if tot:
        fl = sum(r["alarm_ends_flagged_removed_pct"] * r["n_caught"] for r in xai_rows if r.get("n_caught")) / tot
        rd = sum(r["alarm_ends_random_removed_pct"] * r["n_caught"] for r in xai_rows if r.get("n_caught")) / tot
        lines.append(f"  ALL          n={tot:3d}  flagged removed: alarm ends {fl:5.1f}%   random removed: {rd:5.1f}%")
    lines += ["", "Defect groups, pooled over foods (caught by colour + texture; share of alarm from colour)"]
    for p in pooled[::-1]:
        cs = "n/a" if np.isnan(p["colour_share"]) else f"{100 * p['colour_share']:.0f}%"
        lines.append(f"  {p['group']:28s} n={p['n_images']:3d}  caught {100 * p['caught_fusion']:5.1f}%  colour {cs}")
    with open(os.path.join(args.out_dir, f"summary_{VERSION}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    zip_path = shutil.make_archive(args.out_dir.rstrip("/"), "zip", args.out_dir)
    log(f"Done. Everything is in {args.out_dir}")
    log(f"All results in one file: {zip_path}")


main()