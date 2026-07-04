import os
import json
import logging
from PIL import Image

# PyTorch Imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("training_pipeline")

# ==========================================
# 1. FUNSD / CORD Dataset Loader for PyTorch
# ==========================================
class DocumentNERDataset(Dataset):
    """
    Custom PyTorch Dataset for loading document layouts in FUNSD or CORD annotation formats.
    Normalizes coordinates to LayoutLM's [0, 1000] scale.
    """
    def __init__(self, annotation_dir, image_dir, tokenizer=None, max_seq_len=512):
        self.annotation_files = [os.path.join(annotation_dir, f) for f in os.listdir(annotation_dir) if f.endswith('.json')]
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Mapping FUNSD entity labels to standard indices
        self.label_map = {"question": 1, "answer": 2, "header": 3, "other": 0}

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        anno_path = self.annotation_files[idx]
        with open(anno_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract filename (assuming image file matches JSON name or metadata)
        img_name = os.path.splitext(os.path.basename(anno_path))[0] + ".png"
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image to read resolution for coordinate scaling
        width, height = 1000, 1000
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                width, height = img.size

        words = []
        bboxes = []
        labels = []

        # Parse FUNSD annotations: "form" list contains boxes and corresponding texts
        for item in data.get("form", []):
            text = item.get("text", "")
            box = item.get("box", [0, 0, 0, 0])  # [x_min, y_min, x_max, y_max]
            label = item.get("label", "other")
            
            # Normalize boxes to [0, 1000] scale for LayoutLM
            norm_box = [
                int(1000 * (box[0] / width)),
                int(1000 * (box[1] / height)),
                int(1000 * (box[2] / width)),
                int(1000 * (box[3] / height))
            ]
            
            # Bound check coordinates
            norm_box = [max(0, min(1000, coord)) for coord in norm_box]
            
            # Tokenize words individually
            sub_words = text.split()
            for w in sub_words:
                words.append(w)
                bboxes.append(norm_box)
                labels.append(self.label_map.get(label, 0))

        # Tokenization & LayoutLM Input Alignment
        if self.tokenizer is not None:
            # Encode tokens
            encoding = self.tokenizer(
                words,
                is_split_into_words=True,
                return_offsets_mapping=True,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt"
            )
            
            # Align labels and bounding boxes to token outputs
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            offset_mapping = encoding["offset_mapping"].squeeze(0)
            
            aligned_labels = []
            aligned_boxes = []
            
            word_idx = -1
            for i, offset in enumerate(offset_mapping):
                if offset[0] == 0 and offset[1] != 0:
                    word_idx += 1
                
                # If subword token, assign original labels/boxes, else pad
                if word_idx >= 0 and word_idx < len(words):
                    aligned_labels.append(labels[word_idx])
                    aligned_boxes.append(bboxes[word_idx])
                else:
                    aligned_labels.append(-100)  # PyTorch ignore index for cross entropy
                    aligned_boxes.append([0, 0, 0, 0])
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "bbox": torch.tensor(aligned_boxes),
                "labels": torch.tensor(aligned_labels)
            }

        return {
            "words": words,
            "bboxes": bboxes,
            "labels": labels
        }

# ==========================================
# 2. PyTorch Training Loop for LayoutLM
# ==========================================
def train_layoutlm_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        bbox = batch["bbox"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass through LayoutLM
        outputs = model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # Backprop
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate training token accuracy (ignoring pad tokens labeled -100)
        predictions = torch.argmax(logits, dim=-1)
        active_accuracy = labels != -100
        
        correct_predictions += torch.sum((predictions == labels) & active_accuracy).item()
        total_predictions += torch.sum(active_accuracy).item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / max(1, total_predictions)
    return avg_loss, accuracy

# ==========================================
# 3. CRNN CTC Loss Training Loop
# ==========================================
def train_crnn_ctc(model, train_loader, optimizer, criterion, device):
    """
    PyTorch training step for CRNN using CTC Loss for transcription.
    """
    model.train()
    total_loss = 0
    
    for batch_images, labels, input_lengths, label_lengths in train_loader:
        # batch_images: [B, C, H, W]
        # labels: concatenated token ids [Sum of label_lengths]
        # input_lengths: length of features output by CNN [B]
        # label_lengths: original length of words [B]
        
        batch_images = batch_images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: yields shape [T, B, nclass]
        outputs = model(batch_images)
        
        # Log-softmax for CTC loss
        log_probs = outputs.log_softmax(2)
        
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

# Verification execution script
if __name__ == "__main__":
    if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
        print("Starting training script validation...")
        try:
            tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
            # Build dummy model for local architecture validation
            model = LayoutLMForTokenClassification.from_pretrained(
                "microsoft/layoutlm-base-uncased", 
                num_labels=4
            )
            print("Successfully loaded LayoutLM tokenizer & model layout.")
        except Exception as e:
            print(f"Hugging Face weights unavailable locally. Skipping pipeline verify: {e}")
    else:
        print("PyTorch / Transformers dependencies unavailable. Run in virtual environment.")
