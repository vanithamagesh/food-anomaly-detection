# goodonly_inspect_v5: classify AND segment every image, at real-time speed.
# Trained on good product only (colour + texture model from the study).
#
# For every image it gives
#   classification : GOOD or DEFECT, the alarm score, and why (colour or texture)
#   segmentation   : a mask of the defective area, saved as PNG, drawn on the photo
# and it times each image so you can see the real-time speed (images per second).
#
# Modes (set in the Settings cell, or on the command line with --mode)
#   visa      : all six VisA foods, every test image (good and defective), with
#               accuracy, precision, recall, F1, AUROC, and segmentation IoU / Dice
#   my_images : train on a folder of YOUR good photos, then inspect a folder of photos
#               and/or a video file (each frame). Optional subfolders good/ and defect/
#               in the test folder give accuracy figures too.
#   camera    : on your own computer only (not Colab): live webcam, press q to stop.

import argparse, csv, glob, json, os, shutil, subprocess, sys, tarfile, time, urllib.request, math, random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import roc_auc_score
from scipy.ndimage import binary_erosion
import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps

VERSION = "v5"
VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
SPLIT_URL = "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"
VISA_FOODS = ["cashew", "chewinggum", "fryum", "pipe_fryum", "macaroni1", "macaroni2"]
NAMES = {"cashew": "Cashew", "chewinggum": "Chewing gum", "fryum": "Fryum", "pipe_fryum": "Pipe fryum",
         "macaroni1": "Macaroni 1", "macaroni2": "Macaroni 2"}
IMG = 256          # size the model sees
SHOW = 512         # size of the saved pictures
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Settings from the notebook cell (if present), else defaults
S = globals()
MODE = S.get("MODE", "visa")
FOODS = S.get("FOODS", VISA_FOODS)
QUICK_TEST = S.get("QUICK_TEST", False)
DRIVE_FOLDER = S.get("DRIVE_FOLDER", "/content/drive/MyDrive/food_defects")
MY_GOOD_FOLDER = S.get("MY_GOOD_FOLDER", None)
MY_TEST_FOLDER = S.get("MY_TEST_FOLDER", None)
MY_VIDEO = S.get("MY_VIDEO", None)
MY_NAME = S.get("MY_NAME", "my_product")
MY_MODEL = S.get("MY_MODEL", None)
FALSE_REJECT = S.get("FALSE_REJECT", 0.05)
SEG_FRACTION = S.get("SEG_FRACTION", 0.5)   # defect area = pixels above the alarm level AND >= this share of the item's peak


def log(m):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


# ---------------------------------------------------------------- model
def gblur(maps, sigma=4.0):
    k = 2 * int(round(4 * sigma)) + 1
    return torchvision.transforms.functional.gaussian_blur(maps.unsqueeze(1), [k, k], [sigma, sigma]).squeeze(1)


def nn_dist(q, bank, chunk=4096):
    return torch.cat([torch.cdist(q[i:i + chunk], bank).min(1).values for i in range(0, len(q), chunk)])


def coreset(x, n, seed=0, d=128):
    if n >= len(x):
        return torch.arange(len(x), device=x.device)
    g = torch.Generator().manual_seed(seed)
    z = x @ (torch.randn(x.shape[1], d, generator=g).to(x.device) / math.sqrt(d))
    sel = [0]
    mind = torch.cdist(z, z[:1]).squeeze(1)
    for _ in range(n - 1):
        i = int(torch.argmax(mind))
        sel.append(i)
        mind = torch.minimum(mind, torch.cdist(z, z[i:i + 1]).squeeze(1))
    return torch.tensor(sel, device=x.device)


def rgb_to_lab(x):
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], device=x.device)
    t = torch.einsum("ij,bjhw->bihw", M, lin) / torch.tensor([0.95047, 1.0, 1.08883], device=x.device).view(1, 3, 1, 1)
    f = torch.where(t > 216 / 24389, t.clamp(min=1e-12) ** (1 / 3), (24389 / 27 * t + 16) / 116)
    return torch.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


class Inspector:
    """Colour + texture model, trained on good product only."""

    def __init__(self, device, pretrained=True, seg_fraction=0.5):
        self.device, self.seg_fraction = device, seg_fraction
        w = None if not pretrained else torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        m = torchvision.models.wide_resnet50_2(weights=w)
        m.layer4, m.fc = torch.nn.Identity(), torch.nn.Identity()
        self.net = m.eval().to(device)

    @torch.no_grad()
    def _tex_feats(self, x_u8):
        x = (x_u8.to(self.device).float() / 255 - MEAN.to(self.device)) / STD.to(self.device)
        m = self.net
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        l2 = m.layer2(m.layer1(x))
        l3 = m.layer3(l2)
        f2 = F.avg_pool2d(l2, 3, 1, 1)
        f3 = F.interpolate(F.avg_pool2d(l3, 3, 1, 1), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], 1)
        B, C, h, w = f.shape
        f = F.adaptive_avg_pool1d(f.permute(0, 2, 3, 1).reshape(B * h * w, 1, C), 1024)
        return f.reshape(B, h * w, 1024), (h, w)

    @torch.no_grad()
    def _col_feats(self, x_u8):
        return F.avg_pool2d(rgb_to_lab(x_u8.to(self.device).float() / 255), 8, 4, 2)   # B,3,64,64

    @torch.no_grad()
    def _raw_maps(self, x_u8):
        ft, (h, w) = self._tex_feats(x_u8)
        B = ft.shape[0]
        t = nn_dist(ft.reshape(-1, 1024), self.tex_bank).reshape(B, 1, h, w)
        lab = self._col_feats(x_u8)
        c = nn_dist(lab.permute(0, 2, 3, 1).reshape(-1, 3), self.col_bank).reshape(B, 1, *lab.shape[-2:])
        up = lambda m: gblur(F.interpolate(m, size=(IMG, IMG), mode="bilinear", align_corners=False)[:, 0])
        return up(t), up(c)

    @torch.no_grad()
    def fit(self, good_u8, calib_u8, false_reject=0.05, bank_cap=16384, seed=0, bs=16):
        feats, labs = [], []
        for i in range(0, len(good_u8), bs):
            f, _ = self._tex_feats(good_u8[i:i + bs])
            feats.append(f.half().cpu())
            labs.append(self._col_feats(good_u8[i:i + bs]).cpu())
        flat = torch.cat(feats).reshape(-1, 1024)
        g = torch.Generator().manual_seed(seed)
        if len(flat) > 200000:
            flat = flat[torch.randperm(len(flat), generator=g)[:200000]]
        flat = flat.to(self.device).float()
        n = int(min(bank_cap, 0.1 * len(good_u8) * feats[0].shape[1], len(flat)))
        self.tex_bank = flat[coreset(flat, n, seed)].contiguous()
        cols = torch.unique(torch.round(torch.cat(labs).permute(0, 2, 3, 1).reshape(-1, 3) * 2) / 2, dim=0)
        if len(cols) > 200000:
            cols = cols[torch.randperm(len(cols), generator=g)[:200000]]
        self.col_bank = cols.to(self.device).float()
        # put both alarms on one scale using good calibration images
        tm, cm = [], []
        for i in range(0, len(calib_u8), bs):
            t, c = self._raw_maps(calib_u8[i:i + bs])
            tm.append(t.flatten(1).max(1).values.cpu())
            cm.append(c.flatten(1).max(1).values.cpu())
        tm, cm = torch.cat(tm).double(), torch.cat(cm).double()
        self.stats = [float(tm.mean()), float(tm.std().clamp(min=1e-6)), float(cm.mean()), float(cm.std().clamp(min=1e-6))]
        calib_scores = torch.maximum((tm - self.stats[0]) / self.stats[1], (cm - self.stats[2]) / self.stats[3])
        self.threshold = float(np.quantile(calib_scores.numpy(), 1 - false_reject))
        return self

    @torch.no_grad()
    def predict(self, x_u8):
        """-> fused map (B,IMG,IMG), score (B,), defect (B,), cause (B,), mask (B,IMG,IMG), colour_part (B,IMG,IMG)"""
        t, c = self._raw_maps(x_u8)
        zt = (t - self.stats[0]) / self.stats[1]
        zc = (c - self.stats[2]) / self.stats[3]
        fused = torch.maximum(zt, zc)
        score = fused.flatten(1).max(1).values
        cause = np.where(zc.flatten(1).max(1).values.cpu().numpy() > zt.flatten(1).max(1).values.cpu().numpy(),
                         "colour", "texture")
        # segmentation: above the alarm level and at least SEG_FRACTION of this item's peak
        seg_level = torch.clamp(score * self.seg_fraction, min=self.threshold).view(-1, 1, 1)
        mask = fused > seg_level
        return (fused.cpu().numpy(), score.cpu().numpy(), (score > self.threshold).cpu().numpy(), cause,
                mask.cpu().numpy(), (zc > zt).cpu().numpy())

    def save(self, path, info):
        torch.save({"tex_bank": self.tex_bank.half().cpu(), "col_bank": self.col_bank.cpu(), "stats": self.stats,
                    "threshold": self.threshold, "info": info, "version": VERSION}, path)

    def load(self, path):
        d = torch.load(path, map_location="cpu")
        self.tex_bank = d["tex_bank"].float().to(self.device)
        self.col_bank = d["col_bank"].float().to(self.device)
        self.stats, self.threshold = d["stats"], d["threshold"]
        return self


# ---------------------------------------------------------------- data
def download(url, dest):
    if os.path.exists(dest):
        return dest
    log(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest + ".part", "wb") as f:
        shutil.copyfileobj(r, f, 1 << 22)
    os.replace(dest + ".part", dest)
    return dest


def find_visa(root, foods):
    for d in [root] + glob.glob(os.path.join(root, "*")) + glob.glob(os.path.join(root, "*", "*")):
        if all(os.path.isdir(os.path.join(d, f, "Data", "Images")) for f in foods):
            return d
    return None


def prepare_visa(data_root, keep_dir, foods):
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(keep_dir, exist_ok=True)
    root = find_visa(data_root, foods)
    if root is None:
        tar_path = os.path.join(keep_dir, "VisA_20220922.tar")
        download(VISA_URL, tar_path)          # skipped if the study notebook already saved it
        log("Unpacking the food folders (about a minute)")
        with tarfile.open(tar_path) as tar:
            for m in tar:
                parts = [p for p in m.name.split("/") if p not in ("", ".")]
                if parts and ".." not in parts and any(p in foods for p in parts[:2]):
                    try:
                        tar.extract(m, data_root, filter="data")
                    except TypeError:
                        tar.extract(m, data_root)
        root = find_visa(data_root, foods)
    split = os.path.join(keep_dir, "visa_1cls_split.csv")
    try:
        download(SPLIT_URL, split)
    except Exception as e:
        log(f"Official split not available ({e}); using a seeded split")
        split = None
    return root, split


def visa_items(root, split, food, seed=0):
    train, test = [], []
    if split and os.path.exists(split):
        for r in csv.DictReader(open(split)):
            if r["object"] != food:
                continue
            img = os.path.join(root, r["image"])
            bad = r["label"] == "anomaly"
            mask = os.path.join(root, r["mask"]) if bad and r.get("mask") else None
            (train if r["split"] == "train" else test).append((img, mask, int(bad)))
    else:
        rng = random.Random(seed)
        good = sorted(glob.glob(os.path.join(root, food, "Data", "Images", "Normal", "*")))
        bad = sorted(glob.glob(os.path.join(root, food, "Data", "Images", "Anomaly", "*")))
        rng.shuffle(good)
        k = min(len(bad), len(good) // 5)
        train = [(p, None, 0) for p in good[k:]]
        test = [(p, None, 0) for p in good[:k]] + [
            (p, os.path.join(root, food, "Data", "Masks", "Anomaly", os.path.splitext(os.path.basename(p))[0] + ".png"), 1)
            for p in bad]
    return train, test


def list_images(folder):
    return sorted(p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True) if p.lower().endswith(EXTS))


def label_from_path(p):
    parts = [s.lower() for s in p.replace("\\", "/").split("/")]
    if any(s in ("good", "ok", "normal") for s in parts[-3:-1]):
        return 0
    if any(s in ("defect", "defective", "bad", "anomaly", "ng") for s in parts[-3:-1]):
        return 1
    return -1      # unknown


def prep(im):
    """One PIL image -> (model input IMG x IMG, display SHOW x SHOW). Every photo, video frame
    and camera frame goes through exactly this, so training and inspection match."""
    im = im.convert("RGB")
    return (np.asarray(im.resize((IMG, IMG), Image.BILINEAR)), np.asarray(im.resize((SHOW, SHOW), Image.BILINEAR)))


def read_rgb(path, size=IMG):
    return prep(Image.open(path))[0 if size == IMG else 1]


def read_mask(path, size=IMG):
    if not path or not os.path.exists(path):
        return None
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.NEAREST)) > 0


def to_tensor(arrs):
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------- pictures
try:
    FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    FONT_S = ImageFont.truetype("DejaVuSans.ttf", 16)
except Exception:
    FONT = FONT_S = ImageFont.load_default()
HEAT = colormaps["inferno"]


def edge(mask):
    return mask & ~binary_erosion(mask, iterations=2)


def render(photo, fused, mask, colour_part, defect, score, cause, threshold, gt=None, title=""):
    """photo: SHOW x SHOW x 3 uint8. Returns a side-by-side picture: photo | alarm map + defect outline."""
    up = lambda a, mode: np.asarray(Image.fromarray(a).resize((SHOW, SHOW), mode))
    f = np.asarray(Image.fromarray(fused.astype(np.float32)).resize((SHOW, SHOW), Image.BILINEAR))
    m = up(mask.astype(np.uint8) * 255, Image.NEAREST) > 127
    cp = up(colour_part.astype(np.uint8) * 255, Image.NEAREST) > 127
    lo, hi = max(threshold * 0.5, 0.0), max(score, threshold * 2.5, threshold + 1)   # scaled to this item's peak
    a = np.clip((f - lo) / (hi - lo), 0, 1)
    heat = (HEAT(a)[..., :3] * 255).astype(np.float32)
    alpha = (0.65 * (a > 0))[..., None]
    right = (photo * (1 - alpha) + heat * alpha).astype(np.uint8)
    # segmentation: fill the defect area, orange = texture, blue = colour
    fill = right.copy()
    fill[m & ~cp] = (0.5 * fill[m & ~cp] + 0.5 * np.array([213, 94, 0])).astype(np.uint8)
    fill[m & cp] = (0.5 * fill[m & cp] + 0.5 * np.array([0, 114, 178])).astype(np.uint8)
    right = fill
    e = edge(m)
    right[e] = (255, 255, 255)
    left = photo.copy()
    if gt is not None and gt.any():
        g = up(gt.astype(np.uint8) * 255, Image.NEAREST) > 127
        left[edge(g)] = (0, 255, 255)
    canvas = Image.new("RGB", (2 * SHOW + 10, SHOW + 70), "white")
    canvas.paste(Image.fromarray(left), (0, 70))
    canvas.paste(Image.fromarray(right), (SHOW + 10, 70))
    d = ImageDraw.Draw(canvas)
    verdict = f"DEFECT ({cause})" if defect else "GOOD"
    d.rectangle([0, 0, 2 * SHOW + 10, 38], fill=(200, 40, 40) if defect else (30, 140, 60))
    d.text((10, 7), f"{verdict}   score {score:.1f}  (alarm level {threshold:.1f})", fill="white", font=FONT)
    d.text((10, 44), title[:60], fill="black", font=FONT_S)
    d.text((SHOW + 20, 44), "alarm map; area: orange=texture, blue=colour", fill="black", font=FONT_S)
    return canvas


# ---------------------------------------------------------------- metrics
def class_metrics(y, pred, score):
    y, pred = np.asarray(y), np.asarray(pred).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    tn, fn = int(((pred == 0) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"n_images": len(y), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "accuracy": (tp + tn) / max(len(y), 1), "precision": prec, "recall_defects_caught": rec,
            "good_rejected": fp / max(fp + tn, 1), "F1": 2 * prec * rec / max(prec + rec, 1e-9),
            "image_AUROC": roc_auc_score(y, score) if 0 < y.sum() < len(y) else float("nan")}


# ---------------------------------------------------------------- runs
def inspect_images(model, items, out_dir, bs=8, timing=None):
    """items: list of (path, gt_mask_path or None, label 0/1/-1). Saves pictures, masks and a CSV."""
    os.makedirs(os.path.join(out_dir, "pictures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    rows, pix_s, pix_y = [], [], []
    inter = union = gt_sum = pr_sum = 0
    times = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        pairs = [prep(Image.open(p)) for p, _, _ in chunk]
        shows = [b for _, b in pairs]
        x = to_tensor([a for a, _ in pairs])
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fused, score, defect, cause, mask, cpart = model.predict(x)
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / len(chunk))
        for j, (p, mp, y) in enumerate(chunk):
            gt = read_mask(mp)
            if y == 1 and gt is None:
                gt_known = False
            else:
                gt_known = y in (0, 1)
                if gt is None:
                    gt = np.zeros((IMG, IMG), bool)
            stem = os.path.splitext(os.path.basename(p))[0]
            sub = {0: "good_", 1: "defect_"}.get(y, "")
            name = f"{sub}{stem}"
            render(shows[j], fused[j], mask[j], cpart[j], bool(defect[j]), float(score[j]), cause[j],
                   model.threshold, gt if gt_known else None, os.path.basename(p)).save(
                os.path.join(out_dir, "pictures", name + ".jpg"), quality=90)
            Image.fromarray(mask[j].astype(np.uint8) * 255).save(os.path.join(out_dir, "masks", name + "_mask.png"))
            row = {"image": p, "true_label": {0: "good", 1: "defect"}.get(y, "unknown"),
                   "predicted": "defect" if defect[j] else "good", "score": round(float(score[j]), 3),
                   "cause": cause[j] if defect[j] else "", "defect_area_pct": round(100 * float(mask[j].mean()), 2)}
            if gt_known:
                ii, uu = int((mask[j] & gt).sum()), int((mask[j] | gt).sum())
                row["IoU"] = round(ii / uu, 3) if uu else (1.0 if y == 0 else 0.0)
                if y == 1:    # overlap is measured on defective images only
                    inter += ii; union += uu; gt_sum += int(gt.sum()); pr_sum += int(mask[j].sum())
                sel = slice(None, None, 4)
                pix_s.append(fused[j][sel, sel].ravel()); pix_y.append(gt[sel, sel].ravel())
            rows.append(row)
    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "true_label", "predicted", "score", "cause", "defect_area_pct", "IoU"])
        w.writeheader()
        w.writerows(rows)
    seg = {}
    if pix_y:
        py, ps = np.concatenate(pix_y), np.concatenate(pix_s)
        seg = {"pixel_AUROC": roc_auc_score(py, ps) if 0 < py.sum() < len(py) else float("nan"),
               "IoU_defects": inter / max(union, 1), "Dice_defects": 2 * inter / max(gt_sum + pr_sum, 1)}
    ms = 1000 * float(np.median(times)) if times else float("nan")
    return rows, seg, ms


def inspect_video(model, video, out_path, every=1):
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer, n, times, flagged = None, 0, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if (n - 1) % every:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small, show = prep(Image.fromarray(rgb))
        x = to_tensor([small])
        t0 = time.perf_counter()
        fused, score, defect, cause, mask, cpart = model.predict(x)
        times.append(time.perf_counter() - t0)
        flagged += int(defect[0])
        pic = np.asarray(render(show, fused[0], mask[0], cpart[0], bool(defect[0]), float(score[0]), cause[0],
                                model.threshold, None, f"frame {n}  {1 / times[-1]:.0f} frames/s"))
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / every, (pic.shape[1], pic.shape[0]))
        writer.write(cv2.cvtColor(pic, cv2.COLOR_RGB2BGR))
    cap.release()
    if writer:
        writer.release()
    log(f"Video: {len(times)} frames inspected, {flagged} flagged, {1 / np.median(times):.1f} frames/s -> {out_path}")


def run_camera(model, cam=0):
    import cv2
    cap = cv2.VideoCapture(cam)
    log("Camera on. Press q in the window to stop.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small, show = prep(Image.fromarray(rgb))
        t0 = time.perf_counter()
        fused, score, defect, cause, mask, cpart = model.predict(to_tensor([small]))
        fps = 1 / (time.perf_counter() - t0)
        pic = np.asarray(render(show, fused[0], mask[0], cpart[0], bool(defect[0]), float(score[0]), cause[0],
                                model.threshold, None, f"live  {fps:.0f} frames/s"))
        cv2.imshow("goodonly inspection", cv2.cvtColor(pic, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=MODE, choices=["visa", "my_images", "camera"])
    ap.add_argument("--foods", default=",".join(FOODS))
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--good", default=MY_GOOD_FOLDER, help="folder of good photos (my_images mode)")
    ap.add_argument("--test", default=MY_TEST_FOLDER, help="folder of photos to inspect")
    ap.add_argument("--video", default=MY_VIDEO)
    ap.add_argument("--name", default=MY_NAME)
    ap.add_argument("--model", default=MY_MODEL, help="saved model (.pt) to reuse instead of training")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--false_reject", type=float, default=FALSE_REJECT)
    ap.add_argument("--seg_fraction", type=float, default=SEG_FRACTION)
    ap.add_argument("--quick", action="store_true", default=QUICK_TEST)
    ap.add_argument("--no_pretrained", action="store_true", help=argparse.SUPPRESS)
    args, _ = ap.parse_known_args()

    colab = "google.colab" in sys.modules or os.path.isdir("/content")
    keep = os.path.join(DRIVE_FOLDER, "data")
    if colab and args.mode != "camera":
        if not os.path.isdir("/content/drive/MyDrive"):
            from google.colab import drive
            drive.mount("/content/drive")
    else:
        keep = args.data_root or "data_goodonly"
    out = args.out_dir or (os.path.join(DRIVE_FOLDER, "results", f"inspect_{VERSION}") if colab else f"inspect_{VERSION}")
    os.makedirs(out, exist_ok=True)
    data_root = args.data_root or ("/content/data" if colab else "data_goodonly")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Running on {'GPU ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}; results in {out}")
    model = Inspector(device, pretrained=not args.no_pretrained, seg_fraction=args.seg_fraction)
    t_start = time.time()

    if args.mode == "camera":
        if not args.model:
            sys.exit("Give --model path/to/model.pt (made by the my_images or visa mode).")
        run_camera(model.load(args.model), args.camera)
        return

    if args.mode == "visa":
        foods = [f for f in args.foods.split(",") if f]
        if args.quick:
            foods = foods[:2]
        root, split = prepare_visa(data_root, keep, foods)
        summary = []
        for f in foods:
            log(f"=== {NAMES.get(f, f)} ===")
            train, test = visa_items(root, split, f)
            random.Random(0).shuffle(train)
            nc = max(5, len(train) // 10)
            calib, train = train[:nc], train[nc:]
            if args.quick:
                train, calib = train[:60], calib[:20]
                test = [t for t in test if t[2] == 0][:20] + [t for t in test if t[2] == 1][:20]
            log(f"  training on {len(train)} good images; alarm level from {len(calib)} good images; "
                f"inspecting {len(test)} test images")
            t0 = time.time()
            load = lambda its: to_tensor([read_rgb(p, IMG) for p, _, _ in its])
            model.fit(load(train), load(calib), args.false_reject)
            os.makedirs(os.path.join(out, "models"), exist_ok=True)
            model.save(os.path.join(out, "models", f"{f}_model.pt"), {"food": f, "n_train": len(train)})
            log(f"  trained in {time.time() - t0:.0f} s; alarm level {model.threshold:.2f}")
            rows, seg, ms = inspect_images(model, test, os.path.join(out, f))
            cm = class_metrics([r["true_label"] == "defect" for r in rows], [r["predicted"] == "defect" for r in rows],
                               [r["score"] for r in rows])
            s = {"food": f, **cm, **seg, "ms_per_image": ms, "images_per_second": 1000 / ms}
            summary.append(s)
            log(f"  accuracy {100 * cm['accuracy']:.1f}%  defects caught {100 * cm['recall_defects_caught']:.1f}%  "
                f"good rejected {100 * cm['good_rejected']:.1f}%  F1 {cm['F1']:.3f}  "
                f"pixel AUROC {seg.get('pixel_AUROC', float('nan')):.3f}  IoU {seg.get('IoU_defects', float('nan')):.3f}  "
                f"{1000 / ms:.0f} images/s")
            with open(os.path.join(out, f"summary_by_food_{VERSION}.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
                w.writeheader()
                w.writerows(summary)
        keys = ["accuracy", "precision", "recall_defects_caught", "good_rejected", "F1", "image_AUROC",
                "pixel_AUROC", "IoU_defects", "Dice_defects", "ms_per_image", "images_per_second"]
        lines = [f"goodonly_inspect_{VERSION}: classification and segmentation of every test image",
                 f"alarm level set so that about {100 * args.false_reject:.0f}% of good calibration images are rejected", ""]
        lines.append("food          " + " ".join(f"{k[:10]:>10s}" for k in keys))
        for s in summary + [{"food": "MEAN", **{k: float(np.nanmean([x.get(k, np.nan) for x in summary])) for k in keys}}]:
            lines.append(f"{s['food']:13s} " + " ".join(f"{s.get(k, float('nan')):10.3f}" for k in keys))
        lines += ["", "Pictures of every image: <food>/pictures/   masks: <food>/masks/   per image: <food>/predictions.csv"]
        open(os.path.join(out, f"summary_{VERSION}.txt"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines))

    else:   # my_images
        if args.model:
            model.load(args.model)
        else:
            if not args.good:
                sys.exit("Set MY_GOOD_FOLDER (a folder of photos of GOOD product) in the Settings cell.")
            goods = list_images(args.good)
            if len(goods) < 10:
                sys.exit(f"Only {len(goods)} good photos found in {args.good}; at least 10 are needed (50+ is better).")
            random.Random(0).shuffle(goods)
            nc = max(5, len(goods) // 10)
            log(f"Training on {len(goods) - nc} good photos; alarm level from {nc}")
            model.fit(to_tensor([read_rgb(p, IMG) for p in goods[nc:]]), to_tensor([read_rgb(p, IMG) for p in goods[:nc]]),
                      args.false_reject)
            path = os.path.join(out, f"{args.name}_model.pt")
            model.save(path, {"food": args.name, "n_train": len(goods) - nc})
            log(f"Model saved: {path}")
        if args.test:
            items = [(p, None, label_from_path(p)) for p in list_images(args.test)]
            rows, seg, ms = inspect_images(model, items, os.path.join(out, args.name))
            n_def = sum(r["predicted"] == "defect" for r in rows)
            lines = [f"{len(rows)} photos inspected, {n_def} flagged as defective, {1000 / ms:.0f} images/s"]
            known = [r for r in rows if r["true_label"] != "unknown"]
            if known and len({r["true_label"] for r in known}) == 2:
                cm = class_metrics([r["true_label"] == "defect" for r in known], [r["predicted"] == "defect" for r in known],
                                   [r["score"] for r in known])
                lines.append("  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in cm.items()))
            open(os.path.join(out, f"summary_{args.name}_{VERSION}.txt"), "w").write("\n".join(lines) + "\n")
            print("\n".join(lines))
        if args.video:
            inspect_video(model, args.video, os.path.join(out, f"{args.name}_video_checked.mp4"))

    if colab:
        z = shutil.make_archive(out.rstrip("/"), "zip", out)
        log(f"All results in one file: {z}")
    log(f"Done in {(time.time() - t_start) / 60:.1f} min")


main()
