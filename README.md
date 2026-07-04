# Xtract: Multi-Modal Document Intelligence & OCR Extraction

Xtract is a premium, high-aesthetic interactive document intelligence sandbox designed to showcase and benchmark state-of-the-art key information extraction (KIE) and optical character recognition (OCR) models. It integrates a **FastAPI** backend with a modern glassmorphic web dashboard to visualize text extractions and coordinate mappings in real-time.

---

## 🚀 Key Features

* **Multi-Model OCR Sandbox**: Compare lightweight recurrent architectures with heavy transformer architectures:
  * **CRNN (CNN + BLSTM + CTC)**: Fast, line-level text recognition (sub-15ms latency).
  * **TrOCR (ViT + GPT-2)**: High-accuracy sequence-to-sequence visual transformer.
  * **LayoutLM (Multimodal Transformer)**: Fuses spatial layout coordinates (2D bounding boxes) and text tokens for advanced Key Information Extraction.
* **PDF & Multi-Page Support**: Upload `.pdf` documents directly. The system automatically renders individual pages to high-resolution images via `PyMuPDF`, processes them sequentially, and aggregates line items and key-value fields.
* **Auto-Deskewing (Orientation Correction)**: Integrated OpenCV preprocessing detects document text slant via pixel projection checks and rotates the document horizontally before running OCR to prevent line-grouping distortions.
* **Multilingual Parsing Engine**: Out-of-the-box support for key synonym mappings across **English, Spanish, French, and German** (e.g., *base imponible*, *IVA*, *endbetrag*, *tva*, *montant total*).
* **Enhanced Table Extraction**: Advanced right-to-left scanning matches final column prices, aligns description details using HSN/SAC numbers as anchors, and filters out non-item rows (such as bank details, legal footnotes, and signatories).
* **Interactive Bounding Box Canvas**: View bounding boxes directly overlayed on the text. Click or hover coordinates to link cards dynamically between the document preview and the sidebar.
* **Arithmetic Validation & History Clear**: Cleanly wipe logs and sweep temporary uploads from disk to prevent storage leaks.

---

## 🔑 Key Information Extraction (KIE)

Xtract focuses on identifying and structuring key data fields from raw document pages rather than just reading unstructured text streams.

### Target Extraction Fields
* **Merchant Details**: Branding, Merchant GSTIN, PAN Number.
* **Invoice Metadata**: Document Type, Date, Invoice/Reference/Quotation Number.
* **Line Items Table**: Dynamic list containing description and total price per item.
* **Financial Totals**: Subtotal, CGST, SGST, IGST, UTGST, Tax (VAT), and final Total Amount.

### Technical Extraction Strategies
1. **Multimodal LayoutLM (Impira QA Pipeline)**:
   Formulates KIE as document question-answering. Queries are passed to LayoutLM along with the page image and bounding box coordinates (e.g., *"What is the total?"*). The model answers by understanding both visual proximity and semantic textual content.
2. **Fallback Heuristics & Geometrics**:
   When GPU acceleration is unavailable, a template matching and geometric layout engine clusters words by checking horizontal boundary overlaps (determining line items) and evaluates regex filters to determine numeric values, dates, and merchant branding.

---

## 📁 Project Structure

```bash
ocr_document_extraction/
├── app.py                  # FastAPI web server, PDF parsing & routing endpoints
├── ocr_pipeline.py         # OpenCV deskewing, EasyOCR, TrOCR/LayoutLM wrappers & spatial layout parser
├── training_pipeline.py    # PyTorch training execution scripts for CORD & FUNSD datasets
├── generate_samples.py     # Script to generate sample invoice/receipt layouts
├── requirements.txt        # Backend python dependencies
├── static/                 # Frontend assets
│   ├── index.html          # Glassmorphic Web UI layout with dynamic metadata grids
│   ├── styles.css          # Glassmorphic CSS design system
│   └── app.js              # Canvas rendering, events, and Chart.js integration
├── samples/                # Sample document images for testing
└── uploads/                # Directory where uploaded documents are stored
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Configure Virtual Environment
Create and activate a Python virtual environment to manage dependencies cleanly:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (PowerShell)
.venv\Scripts\Activate.ps1

# Activate virtual environment (Command Prompt)
.venv\Scripts\activate.bat

# Activate virtual environment (Bash)
source .venv/bin/activate
```

### 3. Install Dependencies
Install all the required neural network, web, and rendering dependencies:

```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

Start the FastAPI application by running:

```bash
# Run using the virtual environment interpreter
python -m uvicorn app:app --reload
```

Once running, navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🧠 Model Specifications

| Model | Architecture | Best Use Case | Avg. Latency | Weights Size |
| :--- | :--- | :--- | :--- | :--- |
| **CRNN** | CNN + Bi-LSTM + CTC | Line-level cropped text blocks | ~12ms | ~45 MB |
| **TrOCR** | ViT Encoder + GPT-2 Decoder | Rotated, skewed, or handwritten text | ~115ms | ~420 MB |
| **LayoutLM** | Text + 2D Coordinates + Visuals | Forms, receipts, tables (KIE) | ~340ms | ~610 MB |

---

## 📄 License
This project is licensed under the MIT License.
