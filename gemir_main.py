import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import math
import random
from PIL import ImageFilter
import torch.utils.data as data
from PIL import Image
import cv2
import os
import gc
import pandas as pd
import numpy as np
import open_clip
import re
from collections import defaultdict
from torchmetrics.retrieval import RetrievalMAP, RetrievalMRR, RetrievalPrecision


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 1.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class GatedFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate(x)

class PatentNet(nn.Module):
    def __init__(self, model, embedding_size, use_text, text_dropout):
        super().__init__()
        original_conv1 = model.visual.conv1
        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        with torch.no_grad():
            new_conv1.weight[:, 0:1, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
        model.visual.conv1 = new_conv1
        self.model = model
        self.use_text = use_text
        in_features = 512
        self.gate = GatedFusion(embedding_size)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(),
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.PReLU(),
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.text_dropout = text_dropout

    def forward(self, x, text=None):
        if self.use_text and text is not None:
            image_features = self.model.encode_image(x)
            if isinstance(text, (list, tuple)):
                text = self.tokenizer(text).to(x.device)
            text_features = self.model.encode_text(text)
            if self.training:
                mask = torch.bernoulli(torch.full((image_features.shape[0], 1), 1 - self.text_dropout, device=x.device))
                text_features = text_features * mask
            gate = self.gate(image_features)
            fused_features = image_features + gate * text_features
            x = self.head(fused_features)
        else:
            x = self.model.encode_image(x)
            x = self.head(x)
        return x


class MultiLabelClassifier(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, features, labels=None):
        logits = self.fc(features)
        return logits


def train(model, classifier_head, loss_func, device, train_loader, optimizer, epoch):
    print('---start training---')
    model.train()
    classifier_head.train()
    total_loss = 0

    for batch_idx, (data, labels, _) in enumerate(tqdm(train_loader)):
        images, descriptions = data
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        embeddings = model(images, text=descriptions)
        logits = classifier_head(embeddings)
        loss = loss_func(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 500 == 0:
            print("Epoch {} Iteration {}: Train Loss = {:.4f}".format(epoch, batch_idx, loss.item()))
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} Average Train Loss: {avg_loss:.4f}")
    return avg_loss


def get_all_embeddings_batch(dataloader, model, device):
    model.eval()
    embeddings_list = []
    labels_list = []
    pids_list = []

    with torch.no_grad():
        for batch_idx, (data, labels, batch_pids) in enumerate(tqdm(dataloader, desc="Computing embeddings")):
            images, descriptions = data
            images = images.to(device)
            embeddings = model(images, text=descriptions)
            embeddings_list.append(embeddings.cpu())
            labels_list.append(labels.cpu())
            pids_list.extend(batch_pids)

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

    all_embeddings = torch.cat(embeddings_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    all_pids = np.array(pids_list, dtype=object)
    return all_embeddings, all_labels, all_pids


def compute_retrieval_metrics_gpu(sim_matrix, target_matrix, query_pids, gallery_pids, top_k=5):
    """
    Compute Precision@K, mAP@K, MRR with unique PID filtering (GPU-based similarity)
    """
    device = sim_matrix.device
    num_queries = sim_matrix.size(0)

    metric_map = RetrievalMAP(top_k=top_k).to(device)
    metric_mrr = RetrievalMRR(top_k=top_k).to(device)
    metric_precision = RetrievalPrecision(top_k=top_k).to(device)
    
    all_preds = []
    all_targets = []
    all_indexes = []
    
    for i in range(num_queries):
        scores = sim_matrix[i]
        sorted_idx = torch.argsort(scores, descending=True).cpu().numpy()

        unique_indices = []
        seen_pids = set()
        for idx in sorted_idx:
            pid = gallery_pids[idx]
            if pid not in seen_pids:
                unique_indices.append(idx)
                seen_pids.add(pid)
            if len(unique_indices) >= top_k:
                break

        unique_indices = torch.tensor(unique_indices, device=device)
        filtered_scores = scores[unique_indices]
        filtered_targets = target_matrix[i, unique_indices].bool()
        
        all_preds.append(filtered_scores)
        all_targets.append(filtered_targets)
        all_indexes.append(torch.full_like(filtered_scores, i, dtype=torch.long))

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    indexes = torch.cat(all_indexes)
    
    res_p = metric_precision(preds, targets, indexes=indexes)
    res_map = metric_map(preds, targets, indexes=indexes)
    res_mrr = metric_mrr(preds, targets, indexes=indexes)

    return res_p.item(), res_map.item(), res_mrr.item()


def calculate_retrieval_metrics_batch(query_loader, gallery_loader, model, device,
                                          batch_size=256, top_k=5):
    """
    Batched GPU retrieval evaluation with unique PID filtering
    """
    print("Computing query embeddings...")
    query_emb, query_lbl, query_pids = get_all_embeddings_batch(query_loader, model, device)

    print("Computing gallery embeddings...")
    gallery_emb, gallery_lbl, gallery_pids = get_all_embeddings_batch(gallery_loader, model, device)

    query_emb = F.normalize(query_emb, dim=1)
    gallery_emb = F.normalize(gallery_emb, dim=1)

    query_lbl = query_lbl.to(device)
    gallery_lbl = gallery_lbl.to(device)

    total_p, total_map, total_mrr = [], [], []

    for i in range(0, query_emb.size(0), batch_size):
        q_emb = query_emb[i:i + batch_size].to(device)
        q_lbl = query_lbl[i:i + batch_size]
        q_pids = query_pids[i:i + batch_size]

        sim_matrix = torch.matmul(q_emb, gallery_emb.to(device).T)
        target_matrix = (torch.matmul(q_lbl.float(), gallery_lbl.T.float()) > 0)

        p_k, map_k, mrr_k = compute_retrieval_metrics_gpu(
            sim_matrix, target_matrix, q_pids, gallery_pids, top_k
        )

        total_p.append(p_k)
        total_map.append(map_k)
        total_mrr.append(mrr_k)

        del sim_matrix, target_matrix
        torch.cuda.empty_cache()

    return {
        f"precision_at_{top_k}": float(np.mean(total_p)),
        f"map_at_{top_k}": float(np.mean(total_map)),
        f"mrr_at_{top_k}": float(np.mean(total_mrr)),
    }


def test(train_set, test_set, model, device, batch_size=256):
    print('---start test---')
    model.eval()
    if isinstance(train_set, torch.utils.data.Dataset):
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        train_loader = train_set
    if isinstance(test_set, torch.utils.data.Dataset):
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        test_loader = test_set

    top_k = 5
    accuracies = calculate_retrieval_metrics_batch(test_loader, train_loader, model, device, batch_size, top_k=top_k)

    print("Test accuracy (P@{}): {:.4f}, (mAP@{}): {:.4f}, (MRR@{}): {:.4f}".format(
        top_k, accuracies[f"precision_at_{top_k}"],
        top_k, accuracies[f"map_at_{top_k}"],
        top_k, accuracies[f"mrr_at_{top_k}"]
    ))
    return accuracies


def valuation(val_db_set, val_query_set, model, device, batch_size=256):
    print('---start validation---')
    model.eval()
    if isinstance(val_db_set, torch.utils.data.Dataset):
        val_db_loader = torch.utils.data.DataLoader(val_db_set, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        val_db_loader = val_db_set
    if isinstance(val_query_set, torch.utils.data.Dataset):
        val_query_loader = torch.utils.data.DataLoader(val_query_set, batch_size=batch_size, shuffle=False,
                                                       num_workers=4)
    else:
        val_query_loader = val_query_set

    top_k = 5
    accuracies = calculate_retrieval_metrics_batch(val_query_loader, val_db_loader, model, device, batch_size,
                                                   top_k=top_k)

    print("Val accuracy (P@{}): {:.4f}, (mAP@{}): {:.4f}, (MRR@{}): {:.4f}".format(
        top_k, accuracies[f"precision_at_{top_k}"],
        top_k, accuracies[f"map_at_{top_k}"],
        top_k, accuracies[f"mrr_at_{top_k}"]
    ))
    return accuracies


def valuation_calc(model, classifier_head, loss_func, device, val_query_loader, epoch):
    model.eval()
    classifier_head.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (data, labels, _) in enumerate(val_query_loader):
            images, descriptions = data
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images, text=descriptions)
            logits = classifier_head(embeddings)
            loss = loss_func(logits, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_query_loader)
    print(f"Epoch {epoch} Validation Loss: {avg_loss:.4f}")
    return avg_loss


def parse_patent_id(filepath):
    filename = os.path.basename(filepath)
    raw_id = filename.split('-')[0]
    if raw_id.startswith('US'):
        return raw_id[2:]
    return raw_id


def make_data_list(phase):
    csv_root = "/home/data/nas/deeppatent2/patent_batch/"
    train_caption_root = "/home/data/nas/deeppatent2/captions_enh/train_dataset_trn/"

    csv_paths = {
        "train": os.path.join(csv_root, "train_dataset_trn_new.csv"),
        "test_query": os.path.join(csv_root, "test_query_dataset_new.csv"),
        "test_db": os.path.join(csv_root, "test_db_dataset_new.csv"),
        "val_query": os.path.join(csv_root, "val_query_dataset_new.csv"),
        "val_db": os.path.join(csv_root, "val_db_dataset_new.csv"),
    }

    if phase not in csv_paths:
        raise ValueError(f"Unknown phase: {phase}")

    img_root = "/home/data/nas/deeppatent2/"
    
    if phase == "train":
        train_csv_files = sorted([
            os.path.join(train_caption_root, f)
            for f in os.listdir(train_caption_root)
            if f.endswith(".csv")
        ])

        if len(train_csv_files) == 0:
            raise FileNotFoundError(f"No csv files found in: {train_caption_root}")

        sample_df = pd.read_csv(train_csv_files[0])
        required_cols = ["filepath", "caption", "class_related_loc"]
        for col in required_cols:
            if col not in sample_df.columns:
                raise KeyError(f"Column '{col}' missing in {train_csv_files[0]}. Found: {list(sample_df.columns)}")

        sample_dict = {}

        for csv_file in train_csv_files:
            df = pd.read_csv(csv_file)

            for col in required_cols:
                if col not in df.columns:
                    raise KeyError(f"Column '{col}' missing in {csv_file}. Found: {list(df.columns)}")

            for _, row in df.iterrows():
                rel_path = str(row["filepath"]).strip()

                if rel_path not in sample_dict:
                    full_path = os.path.join(img_root, rel_path)
                    pid = parse_patent_id(rel_path)

                    loc_str = str(row["class_related_loc"]).strip()
                    if loc_str.lower() == "nan" or loc_str == "":
                        labels = []
                    else:
                        raw = [x.strip() for x in loc_str.replace(";", " ").split()]
                        labels = sorted(list(set(int(x) for x in raw if x.isdigit())))

                    sample_dict[rel_path] = {
                        "full_path": full_path,
                        "labels": labels,
                        "captions": [],
                        "pid": pid,
                    }

                caption = str(row["caption"]).strip()
                if caption.lower() == "nan" or caption == "":
                    caption = " "
                sample_dict[rel_path]["captions"].append(caption)

        path_list = []
        desc_list = []
        label_list = []
        pid_list = []

        for rel_path, item in sample_dict.items():
            path_list.append(item["full_path"])
            desc_list.append(item["captions"])
            label_list.append(item["labels"])
            pid_list.append(item["pid"])

        print(f"[{phase}] Loaded {len(path_list)} unique samples from {len(train_csv_files)} csv files in {train_caption_root}")
        return path_list, label_list, desc_list, pid_list

    else:
        csv_file = csv_paths[phase]
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        df = pd.read_csv(csv_file)

        required_cols = ["filepath", "caption", "class_related_loc"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' missing in {csv_file}. Found: {list(df.columns)}")

        path_list = []
        desc_list = []
        label_list = []
        pid_list = []

        for _, row in df.iterrows():
            rel_path = str(row["filepath"]).strip()
            full_path = os.path.join(img_root, rel_path)
            path_list.append(full_path)

            pid = parse_patent_id(rel_path)
            pid_list.append(pid)

            caption = str(row["caption"]).strip()
            if caption.lower() == "nan" or caption == "":
                caption = " "
            desc_list.append(caption)

            loc_str = str(row["class_related_loc"]).strip()
            if loc_str.lower() == "nan" or loc_str == "":
                labels = []
            else:
                raw = [x.strip() for x in loc_str.replace(";", " ").split()]
                labels = sorted(list(set(int(x) for x in raw if x.isdigit())))
            label_list.append(labels)

        print(f"[{phase}] Loaded {len(path_list)} samples from {csv_file}")
        return path_list, label_list, desc_list, pid_list


def cv2pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


class PatentDataset(data.Dataset):
    def __init__(self, file_list, label_list, desc_list, pid_list, transform, label_map):
        self.file_list = file_list
        self.label_list = label_list
        self.desc_list = desc_list
        self.pid_list = pid_list
        self.transform = transform
        self.label_map = label_map
        self.num_classes = len(label_map)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((224, 224), dtype=np.uint8)
    
        _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            x, y, w, h = cv2.boundingRect(cnt)
            img = img[y:y + h, x:x + w]
        img = cv2pil(img)
        img = self.transform(img)
    
        desc = self.desc_list[index]
        if isinstance(desc, list):
            if len(desc) > 0:
                desc = random.choice(desc)
            else:
                desc = " "
    
        raw_labels = self.label_list[index]
        pid = self.pid_list[index]
    
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        for raw_id in raw_labels:
            if raw_id in self.label_map:
                mapped_idx = self.label_map[raw_id]
                label_vec[mapped_idx] = 1.0
    
        return (img, desc), label_vec, pid


class AsymmetricLossOptimized(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        self.targets = y
        self.anti_targets = 1 - y

        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)

            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.mean()


if __name__ == '__main__':
    seed_everything(42)
    # params
    device = torch.device("cuda")
    batch_size = 256
    num_workers = 4
    eval_batch_size = 256
    use_text = True

    train_list, train_label_list, train_desc_list, train_pid_list = make_data_list(phase="train")
    test_query_list, test_query_label_list, test_query_desc_list, test_query_pid_list = make_data_list(
    phase="test_query")
    test_db_list, test_db_label_list, test_db_desc_list, test_db_pid_list = make_data_list(phase="test_db")
    val_query_list, val_query_label_list, val_query_desc_list, val_query_pid_list = make_data_list(phase="val_query")
    val_db_list, val_db_label_list, val_db_desc_list, val_db_pid_list = make_data_list(phase="val_db")

    all_labels = set()
    for dataset_labels in [train_label_list, val_db_label_list, val_query_label_list, test_db_label_list,
                           test_query_label_list]:
        for sublist in dataset_labels:
            for lbl in sublist:
                all_labels.add(lbl)

    sorted_unique_labels = sorted(list(all_labels))
    num_classes = len(sorted_unique_labels)
    print("num_classes: ", num_classes)
    label_map = {original_id: idx for idx, original_id in enumerate(sorted_unique_labels)}

    data_transform = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size=224),
            transforms.RandomCrop((224), pad_if_needed=True, fill=255),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.22, 0.33), ratio=(0.3, 3.3), value=1, inplace=False),
            transforms.Normalize((0.5,), (0.5,)),
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size=224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    }

    train_data = PatentDataset(train_list, train_label_list, train_desc_list, train_pid_list, data_transform['train'], label_map)
    test_query_data = PatentDataset(test_query_list, test_query_label_list, test_query_desc_list, test_query_pid_list, data_transform['val'], label_map)
    test_db_data = PatentDataset(test_db_list, test_db_label_list, test_db_desc_list, test_db_pid_list, data_transform['val'], label_map)
    val_query_data = PatentDataset(val_query_list, val_query_label_list, val_query_desc_list, val_query_pid_list, data_transform['val'], label_map)
    val_db_data = PatentDataset(val_db_list, val_db_label_list, val_db_desc_list, val_db_pid_list, data_transform['val'], label_map)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_query_loader = torch.utils.data.DataLoader(val_query_data, batch_size=batch_size, num_workers=num_workers, drop_last=False)

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
    embedding_size = 512
    
    model = PatentNet(model, embedding_size=embedding_size, use_text=use_text, text_dropout=0.0).to(device)
    model = nn.DataParallel(model)
    
    loss_func = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=1, clip=0.05).to(device)
    classifier_head = MultiLabelClassifier(embedding_size, num_classes).to(device)

    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': 5e-6, 'weight_decay': 1e-4},
        {'params': classifier_head.parameters(), 'lr': 5e-5, 'weight_decay': 1e-4}
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_val_map = 0.0
    best_epoch = 0
    
    patience = 5
    patience_counter = 0
    
    checkpoint_dir = '/home/data/DesignCLIP-main/checkpoints_impressionclip_v2'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoints_path = "/home/data/DesignCLIP-main/checkpoints_impressionclip_v1/model_epoch_12.pth"
    pretrained_model = torch.load(checkpoints_path, map_location=device)
    
    model.load_state_dict(pretrained_model['model_state_dict'])
    classifier_head.load_state_dict(pretrained_model['classifier_head_dict'])
    
    history = {'train_loss': [], 'val_loss': [], 'val_map': [], 'test_map': []}
    num_epochs = 3
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 60}\nEpoch {epoch}/{num_epochs}\n{'=' * 60}")
        
        train_loss = train(model, classifier_head, loss_func, device, train_loader, optimizer, epoch)
        history['train_loss'].append(train_loss)
        val_loss = valuation_calc(model, classifier_head, loss_func, device, val_query_loader, epoch)
        history['val_loss'].append(val_loss)

        val_accuracies = valuation(val_db_data, val_query_data, model, device, batch_size=eval_batch_size)
        val_map = val_accuracies["map_at_5"]
        history['val_map'].append(val_map)

        if epoch % 1 == 0:
            print("\n--- Test Set Evaluation ---")
            test_accuracies = test(test_db_data, test_query_data, model, device, batch_size=eval_batch_size)
            history['test_map'].append(test_accuracies["map_at_5"])
        else:
            history['test_map'].append(None)

        scheduler.step(val_map)
        epoch_save_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'classifier_head_dict': classifier_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_map': val_map,
        }, epoch_save_path)
        print(f"Saved checkpoint for epoch {epoch} to {epoch_save_path}")

        if val_map > best_val_map:
            best_val_map = val_map
            best_epoch = epoch
            patience_counter = 0
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'classifier_head_dict': classifier_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_map': val_map,
            }, best_model_path)
            print(f"New best model saved! mAP@5 (Unique IDs): {val_map:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training completed. Loading best model for final test.")
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test(test_db_data, test_query_data, model, device, batch_size=eval_batch_size)
