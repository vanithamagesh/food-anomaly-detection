# goodonly_figures_v7 (v7.1: zoom on multi-piece foods, single-type overview): publication figures from the ORIGINAL full-resolution photos.
# Method = the paper's main method (colour + texture, light ResNet-18), same settings as v6.
# Output (600 dpi PNG + TIFF): one figure per food (4 caught defects of different types + 1 good item)
# and an overview figure (one example per food).
import csv, glob, math, os, random, shutil, sys, tarfile, time, urllib.request
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_erosion, binary_dilation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch

S = globals()
DRIVE_FOLDER = S.get("DRIVE_FOLDER", "/content/drive/MyDrive/food_defects")
FOODS = S.get("FOODS", ["cashew", "chewinggum", "fryum", "pipe_fryum", "macaroni1", "macaroni2"])
OUT_DIR = S.get("OUT_DIR_FIGURES") or os.path.join(DRIVE_FOLDER, "results", "figures_final")
DATA_ROOT = S.get("DATA_ROOT", "/content/data")
PRETRAINED = S.get("PRETRAINED", True)
N_DEFECTS = S.get("N_DEFECTS", 4)
# colour margins chosen on fake defects in the v6 run (light version)
MARGIN = S.get("MARGIN", {"cashew": 3.0, "chewinggum": 1.0, "fryum": 0.0, "pipe_fryum": 3.0,
                          "macaroni1": 3.0, "macaroni2": 0.5})
HIRES = {"macaroni1", "macaroni2"}
FALSE_REJECT, SEED = 0.05, 0
VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
SPLIT_URL = "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"
NAMES = {"cashew": "Cashew", "chewinggum": "Chewing gum", "fryum": "Fryum", "pipe_fryum": "Pipe fryum",
         "macaroni1": "Macaroni 1", "macaroni2": "Macaroni 2"}
VISA_ID2CLASS = {
    "fryum": {1: "Burnt", 2: "Corner or Edge Breakage", 3: "Different Colour Spot", 4: "Fryum Stuck Together",
              5: "Middle Breakage", 6: "Similar Colour Spot", 7: "Small Scratches", 8: "Other"},
    "cashew": {1: "Burnt", 2: "Corner or Edge Breakage", 3: "Different Colour Spot", 4: "Middle Breakage",
               5: "Small Holes", 6: "Small Scratches", 7: "Stuck Together", 8: "Same Colour Spot", 9: "Other"},
    "chewinggum": {1: "Chunk of gum missing", 2: "Corner Missing", 3: "Scratches", 4: "Similar Colour Spot",
                   5: "Small Cracks", 6: "Other"},
    "macaroni1": {1: "Chip Around Edge And Corner", 2: "Different Colour Spot", 3: "Middle Breakage",
                  4: "Similar Colour Spot", 5: "Small Cracks", 6: "Small Scratches", 7: "Other"},
    "macaroni2": {1: "Breakage down the middle", 2: "Color spot similar to the Object", 3: "Different Color spot",
                  4: "Small chip around edge", 5: "Small Cracks", 6: "Small Scratches", 7: "Other"},
    "pipe_fryum": {1: "Burnt", 2: "Corner And Edge Breakage", 3: "Different Colour Spot", 4: "Middle Breakage",
                   5: "Similar Colour Spot", 6: "Small Scratches", 7: "Stuck Together", 8: "Small Cracks", 9: "Other"},
}
TEXTURE_RGB = np.array([213, 94, 0]) / 255.0
COLOUR_RGB = np.array([0, 114, 178]) / 255.0
OUTLINE_RGB = np.array([255, 215, 0]) / 255.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def log(m):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


def group(name):
    n = name.lower()
    for key, g in (("burnt", "Burnt"), ("stuck", "Stuck together"), ("scratch", "Scratches"), ("crack", "Cracks"),
                   ("hole", "Holes")):
        if key in n:
            return g
    if ("similar" in n or "same" in n) and ("colo" in n or "spot" in n):
        return "Spot of similar colour"
    if "different" in n and "colo" in n:
        return "Spot of different colour"
    if "missing" in n or "chunk" in n:
        return "Missing piece"
    if "break" in n or "chip" in n or "corner" in n or "edge" in n:
        return "Broken piece"
    return "Other"


# ------------------------------------------------------------------ data
def find_root(base):
    for d in [base] + glob.glob(os.path.join(base, "*")) + glob.glob(os.path.join(base, "*", "*")):
        if os.path.isdir(os.path.join(d, FOODS[0], "Data", "Images")):
            return d
    return None


def prepare_data():
    root = find_root(DATA_ROOT)
    keep = os.path.join(DRIVE_FOLDER, "data")
    if root is None:
        tar_path = os.path.join(keep, "VisA_20220922.tar")
        if not os.path.exists(tar_path):
            os.makedirs(keep, exist_ok=True)
            log("Downloading VisA (first time only)")
            urllib.request.urlretrieve(VISA_URL, tar_path)
        log("Unpacking the food folders")
        os.makedirs(DATA_ROOT, exist_ok=True)
        with tarfile.open(tar_path) as tar:
            for m in tar:
                parts = [p for p in m.name.split("/") if p not in ("", ".")]
                if parts and ".." not in parts and any(p in FOODS for p in parts[:2]):
                    try:
                        tar.extract(m, DATA_ROOT, filter="data")
                    except TypeError:
                        tar.extract(m, DATA_ROOT)
        root = find_root(DATA_ROOT)
    split = os.path.join(keep, "visa_1cls_split.csv")
    if not os.path.exists(split):
        os.makedirs(keep, exist_ok=True)
        urllib.request.urlretrieve(SPLIT_URL, split)
    return root, split


def items(root, split, food):
    train, test = [], []
    for r in csv.DictReader(open(split)):
        if r["object"] != food:
            continue
        img = os.path.join(root, r["image"])
        bad = r["label"] == "anomaly"
        mask = os.path.join(root, r["mask"]) if bad and r.get("mask") else None
        (train if r["split"] == "train" else test).append((img, mask, int(bad)))
    return train, test


def load_small(path, size):
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)


# ------------------------------------------------------------------ model (same as v6, light version)
class Backbone(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet18(weights=w)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.l1, self.l2, self.l3 = m.layer1, m.layer2, m.layer3
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        f2 = self.l2(self.l1(self.stem(x)))
        return f2, self.l3(f2)


def nn_dist(q, bank, qc=4096):
    return torch.cat([torch.cdist(q[i:i + qc], bank).min(1).values for i in range(0, len(q), qc)])


def coreset(x, k, dev, seed=0, d=128):
    n = x.shape[0]
    if k >= n:
        return torch.arange(n)
    g = torch.Generator().manual_seed(seed)
    P = (torch.randn(x.shape[1], d, generator=g) / math.sqrt(d)).to(dev)
    z = torch.cat([x[i:i + 65536].to(dev).float() @ P for i in range(0, n, 65536)])
    zn = (z * z).sum(1)
    s = int(torch.randint(n, (1,), generator=g).item())
    sel = torch.empty(k, dtype=torch.long, device=dev)
    sel[0] = s
    mind = (zn - 2 * (z @ z[s]) + zn[s]).clamp_min(0)
    for t in range(1, k):
        j = torch.argmax(mind)
        sel[t] = j
        mind = torch.minimum(mind, (zn - 2 * (z @ z[j]) + zn[j]).clamp_min(0))
    return sel.cpu()


class Texture:
    def __init__(self, bb, dev, bank_cap=10000, max_patches=1024):
        self.bb, self.dev, self.cap, self.mp = bb, dev, bank_cap, max_patches

    @torch.no_grad()
    def embed(self, imgs, sample=False, bs=16):
        out, g = [], torch.Generator().manual_seed(SEED)
        pool = nn.AvgPool2d(3, 1, 1)
        for i in range(0, len(imgs), bs):
            x = torch.from_numpy(imgs[i:i + bs]).permute(0, 3, 1, 2).float().div(255)
            x = ((x - MEAN) / STD).to(self.dev)
            f2, f3 = self.bb(x)
            f2, f3 = pool(f2), pool(f3)
            e = torch.cat([f2, F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)], 1)
            b, c, h, w = e.shape
            e = e.permute(0, 2, 3, 1).reshape(b, h * w, c)
            if sample and h * w > self.mp:
                keep = torch.stack([torch.randperm(h * w, generator=g)[: self.mp] for _ in range(b)]).to(e.device)
                e = torch.gather(e, 1, keep.unsqueeze(-1).expand(-1, -1, c))
            out.append(e.half().cpu())
            self.grid = (h, w)
        return torch.cat(out)

    def fit(self, imgs):
        x = self.embed(imgs, sample=True).reshape(-1, 384)
        if len(x) > 1000000:
            x = x[torch.randperm(len(x), generator=torch.Generator().manual_seed(SEED))[:1000000]]
        k = int(min(max(1, round(0.1 * len(x))), self.cap))
        self.bank = x[coreset(x, k, self.dev)].to(self.dev).float()

    @torch.no_grad()
    def score(self, imgs, chunk=16):
        maps = []
        for i in range(0, len(imgs), chunk):
            e = self.embed(imgs[i:i + chunk])
            h, w = self.grid
            for j in range(len(e)):
                maps.append(nn_dist(e[j].to(self.dev).float(), self.bank).reshape(h, w).cpu().numpy())
        return np.stack(maps)


def rgb_to_lab(x):
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], device=x.device)
    t = torch.einsum("ij,bjhw->bihw", M, lin) / torch.tensor([0.95047, 1.0, 1.08883], device=x.device).view(1, 3, 1, 1)
    f = torch.where(t > 216 / 24389, t.clamp(min=1e-12) ** (1 / 3), (24389 / 27 * t + 16) / 116)
    return torch.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


class Colour:
    def __init__(self, dev, size):
        self.dev, self.grid = dev, max(4, size // 8)

    @torch.no_grad()
    def embed(self, imgs, bs=32):
        out, cell = [], imgs.shape[1] // self.grid
        for i in range(0, len(imgs), bs):
            x = torch.from_numpy(imgs[i:i + bs][:, : cell * self.grid, : cell * self.grid]).to(self.dev)
            lab = F.avg_pool2d(rgb_to_lab(x.permute(0, 3, 1, 2).float() / 255), cell)
            out.append(lab.permute(0, 2, 3, 1).cpu())
        return torch.cat(out)

    @staticmethod
    def idx(lab):
        L = lab[..., 0].round().clamp(0, 100).long()
        a = (lab[..., 1].round() + 128).clamp(0, 255).long()
        b = (lab[..., 2].round() + 128).clamp(0, 255).long()
        return (L * 256 + a) * 256 + b

    def fit(self, imgs):
        counts = np.bincount(self.idx(self.embed(imgs).reshape(-1, 3)).numpy(), minlength=101 * 256 * 256)
        self.table = torch.from_numpy(distance_transform_edt(~(counts >= 1).reshape(101, 256, 256))
                                      .astype(np.float32).ravel()).to(self.dev)

    def score(self, imgs):
        return self.table[self.idx(self.embed(imgs)).to(self.dev)].cpu().numpy()


def finish(maps, size, sigma):
    t = torch.from_numpy(np.ascontiguousarray(maps)).float().unsqueeze(1)
    up = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[:, 0].numpy()
    return np.stack([gaussian_filter(m, sigma) for m in up]).astype(np.float32)


# ------------------------------------------------------------------ figure helpers
def setup_font():
    path = None
    for p in (os.path.join(DRIVE_FOLDER, "times.ttf"), "/content/times.ttf"):
        if os.path.exists(p):
            path = p
    name = "DejaVu Serif"
    if path:
        fm.fontManager.addfont(path)
        name = fm.FontProperties(fname=path).get_name()
    else:
        lib = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
        if not os.path.exists(lib):
            os.system("apt-get -qq install -y fonts-liberation > /dev/null 2>&1")
        if os.path.exists(lib):
            fm.fontManager.addfont(lib)
            name = "Liberation Serif"
    plt.rcParams.update({"font.family": name, "font.size": 8, "pdf.fonttype": 42, "savefig.dpi": 600})
    log(f"Figure font: {name}")


def crop_box(img, full_image):
    """Square crop around the product (whole frame for multi-piece foods)."""
    H, W = img.shape[:2]
    if full_image:
        return 0, 0, None          # whole frame, original shape
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).astype(np.float32)
    fg = np.abs(img.astype(np.float32) - np.median(border, 0)).sum(2) > 60
    if fg.mean() < 0.01:
        s = min(H, W)
        return (H - s) // 2, (W - s) // 2, s
    ys, xs = np.nonzero(fg)
    y0, y1 = np.percentile(ys, [0.5, 99.5])
    x0, x1 = np.percentile(xs, [0.5, 99.5])
    s = int(max(y1 - y0, x1 - x0) * 1.15)
    s = min(max(s, 64), H, W)
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    top = int(np.clip(cy - s / 2, 0, H - s))
    left = int(np.clip(cx - s / 2, 0, W - s))
    return top, left, s


def outline(mask, width):
    return mask & ~binary_erosion(mask, iterations=max(1, width))


def render_tiles(ex, thr, full_image, tile=900):
    """ex: dict with path, mask_path, fused (model grid), colour_src (bool). Returns 3 tiles."""
    img = np.asarray(Image.open(ex["path"]).convert("RGB"))
    H, W = img.shape[:2]
    up = lambda m, mode: np.asarray(Image.fromarray(m).resize((W, H), mode))
    fused = up(ex["fused"].astype(np.float32), Image.BILINEAR)
    src = up(ex["colour_src"].astype(np.uint8) * 255, Image.NEAREST) > 127
    gt = np.zeros((H, W), bool)
    if ex["mask_path"] and os.path.exists(ex["mask_path"]):
        a = np.asarray(Image.open(ex["mask_path"]))
        a = a[..., 0] if a.ndim == 3 else a
        gt = np.asarray(Image.fromarray((a > 0).astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
    t, l, s = crop_box(img, full_image)
    if s is None:   # several pieces per photo: zoom on the defective piece (or the strongest alarm)
        s = int(0.5 * min(H, W))
        if gt.any():
            ys, xs = np.nonzero(gt)
        else:
            ys, xs = np.unravel_index(np.argmax(fused), fused.shape)
            ys, xs = np.array([ys]), np.array([xs])
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        t, l = int(np.clip(cy - s // 2, 0, H - s)), int(np.clip(cx - s // 2, 0, W - s))
    sl = (slice(t, t + s), slice(l, l + s))
    img, fused, src, gt = img[sl] / 255.0, fused[sl], src[sl], gt[sl]
    lw = max(2, s // 220)
    # 1 photo + inspector outline
    a = img.copy()
    if gt.any():
        a[binary_dilation(outline(gt, lw), iterations=1)] = OUTLINE_RGB
    # 2 alarm map: only where above the good-product level, scaled to the item's peak
    lo, hi = 0.5 * thr, max(thr * 2.5, float(fused.max()))
    v = np.clip((fused - lo) / (hi - lo + 1e-9), 0, 1)
    heat = plt.get_cmap("inferno")(v)[..., :3]
    alpha = (0.7 * np.clip(v * 1.6, 0, 1))[..., None]
    b = img * (1 - alpha) + heat * alpha
    # 3 why flagged: flagged pixels coloured by cause, the rest grey
    flag = fused >= thr
    gray = np.repeat(img.mean(2, keepdims=True), 3, 2) * 0.8
    c = gray.copy()
    c[flag & ~src] = 0.3 * c[flag & ~src] + 0.7 * TEXTURE_RGB
    c[flag & src] = 0.3 * c[flag & src] + 0.7 * COLOUR_RGB
    if flag.any():
        c[outline(flag, lw)] = 1.0
    th = int(round(tile * a.shape[0] / a.shape[1]))
    res = lambda x: np.asarray(Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)).resize((tile, th), Image.LANCZOS))
    return res(a), res(b), res(c)


def draw(columns, titles, path, width_cm=17.4, col_label=True):
    rows = ["Input image\n(ground truth)", "Anomaly map", "Evidence\nattribution"]
    n = len(columns)
    w_in = width_cm / 2.54
    cell = min((w_in - 0.45) / n, 1.7)
    aspect = columns[0][0].shape[0] / columns[0][0].shape[1]
    w_in = min(w_in, cell * n + 0.45)
    fig, axes = plt.subplots(3, n, figsize=(w_in, cell * aspect * 3 + 0.75), squeeze=False)
    for j, (tiles, title) in enumerate(zip(columns, titles)):
        for i in range(3):
            ax = axes[i, j]
            ax.imshow(tiles[i], interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
        if col_label:
            axes[0, j].set_title(title, fontsize=7.5, pad=3)
    for i in range(3):
        axes[i, 0].set_ylabel(rows[i], fontsize=7.5, labelpad=3)
    fig.legend(handles=[Patch(color=OUTLINE_RGB, label="Ground-truth defect"),
                        Patch(color=TEXTURE_RGB, label="Texture evidence"),
                        Patch(color=COLOUR_RGB, label="Colour evidence (CIELAB)")],
               loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.075, wspace=0.03, hspace=0.03)
    for ext in ("png", "tiff"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        fig.savefig(f"{path}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.04, **kw)
    plt.close(fig)
    log(f"  saved {os.path.basename(path)}.png / .tiff")


# ------------------------------------------------------------------ main
def main():
    if os.path.isdir("/content") and not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_font()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Running on {dev}")
    root, split = prepare_data()
    bb = Backbone(PRETRAINED).to(dev)
    overview, overview_titles, notes = [], [], []
    for food in FOODS:
        log(f"=== {NAMES[food]} ===")
        size = 512 if food in HIRES else 256
        sigma = 4.0 * size / 256
        train, test = items(root, split, food)
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(train))
        n_val = min(max(10, int(round(0.1 * len(train)))), max(1, len(train) // 3))
        val = [train[i] for i in idx[:n_val]]
        train = [train[i] for i in idx[n_val:]]
        Xtr = np.stack([load_small(p, size) for p, _, _ in train])
        Xva = np.stack([load_small(p, size) for p, _, _ in val])
        bad = [t for t in test if t[2] == 1]
        good = [t for t in test if t[2] == 0]
        Xb = np.stack([load_small(p, size) for p, _, _ in bad])
        Xg = np.stack([load_small(p, size) for p, _, _ in good])
        tex, col = Texture(bb, dev), Colour(dev, size)
        tex.fit(Xtr)
        col.fit(Xtr)
        del Xtr
        tv, cv = finish(tex.score(Xva), size, sigma), finish(col.score(Xva), size, sigma)
        tvm, cvm = tv.reshape(len(tv), -1).max(1), cv.reshape(len(cv), -1).max(1)
        mt, st, mc, sc = tvm.mean(), tvm.std() + 1e-8, cvm.mean(), cvm.std() + 1e-8
        w = MARGIN.get(food, 0.0)
        thr = float(np.quantile(np.maximum((tvm - mt) / st, (cvm - mc) / sc - w), 1 - FALSE_REJECT))

        def fused_of(X):
            zt = (finish(tex.score(X), size, sigma) - mt) / st
            zc = (finish(col.score(X), size, sigma) - mc) / sc - w
            return np.maximum(zt, zc), zc > zt

        fb, sb = fused_of(Xb)
        fg_, sg = fused_of(Xg)
        scb, scg = fb.reshape(len(fb), -1).max(1), fg_.reshape(len(fg_), -1).max(1)
        caught = scb >= thr
        log(f"  defects caught {100 * caught.mean():.0f}%, good passed {100 * (scg < thr).mean():.0f}% "
            f"(colour margin {w:g}, alarm level {thr:.2f})")
        # defect group of each defective image (from mask class ids)
        groups = []
        for p, mp, _ in bad:
            g = "Other"
            if mp and os.path.exists(mp):
                a = np.asarray(Image.open(mp))
                a = a[..., 0] if a.ndim == 3 else a
                ids = [int(v) for v in np.unique(a) if v]
                names = sorted({group(VISA_ID2CLASS.get(food, {}).get(i, "Other")) for i in ids})
                g = names[0] if len(names) == 1 else ("Several defects" if names else "Other")
            groups.append(g)
        # pick caught defects of different groups (biggest groups first), median score within group
        chosen = []
        by_group = {}
        for i in np.nonzero(caught)[0]:
            by_group.setdefault(groups[i], []).append(i)
        for g in sorted(by_group, key=lambda k: -len(by_group[k])):
            if g in ("Other", "Several defects") and len(by_group) > N_DEFECTS:
                continue
            ids = sorted(by_group[g], key=lambda i: scb[i])
            chosen.append(ids[len(ids) // 2])
            if len(chosen) == N_DEFECTS:
                break
        full_image = food in HIRES
        cols, titles = [], []
        for i in chosen:
            cols.append(render_tiles({"path": bad[i][0], "mask_path": bad[i][1], "fused": fb[i],
                                      "colour_src": sb[i]}, thr, full_image))
            titles.append(groups[i])
        passed = np.nonzero(scg < thr)[0]
        if len(passed):
            k = passed[np.argsort(scg[passed])[len(passed) // 2]]
            cols.append(render_tiles({"path": good[k][0], "mask_path": None, "fused": fg_[k],
                                      "colour_src": sg[k]}, thr, full_image))
            titles.append("Normal (accepted)")
        draw(cols, titles, os.path.join(OUT_DIR, f"fig_real_{food}"))
        notes.append(f"{NAMES[food]}: caught {100 * caught.mean():.0f}% of defects, passed "
                     f"{100 * (scg < thr).mean():.0f}% of good items; columns: " + ", ".join(titles))
        single = [k for k, t_ in enumerate(titles[:len(chosen)]) if t_ not in ("Several defects", "Other")]
        if chosen:
            k = single[0] if single else 0
            overview.append(cols[k])
            overview_titles.append(f"{NAMES[food]}\n{titles[k]}")
        del tex, col, Xb, Xg, fb, fg_
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if overview:
        draw(overview, overview_titles, os.path.join(OUT_DIR, "fig_real_overview"))
    open(os.path.join(OUT_DIR, "figures_final_notes.txt"), "w").write(
        "Figures from the original full-resolution photos; method: CT-R18 (colour-texture fusion, ResNet-18),\n"
        "colour margins and alarm level as in v6. Each defect column is a caught item with the median\n"
        "score of its defect group; the last column is a good item the system passed (median score).\n\n"
        + "\n".join(notes) + "\n")
    z = shutil.make_archive(OUT_DIR.rstrip("/"), "zip", OUT_DIR)
    log(f"Done. Figures in {OUT_DIR}  (zip: {z})")


main()
