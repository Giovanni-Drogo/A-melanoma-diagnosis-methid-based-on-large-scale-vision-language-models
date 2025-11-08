# CLIP-Based Melanoma Diagnosis (Reproduction)

🔬 This project is a reproduction of the paper **"A Melanoma Diagnosis Method Based on Large-Scale Vision-Language Models"** published in *Acta Anatomica Sinica* (2025). The original work proposes a CLIP-based framework enhanced with clinical prompt guidance and partial image encoder fine-tuning for melanoma classification from dermoscopic images.

---

## 📁 Project Structure
.
├── main.py # Main training entry (full model)
├── main_woIft.py # Ablation: Baseline + Clinical Guidance (w/o IFT)
├── main_wocg.py # Ablation: Baseline + IFT (w/o Clinical Guidance)
├── main_woIft_cg.py # Ablation: Baseline (CoOp only)
├── main_kgcoop.py # Comparison: KgCoOp method
├── test.py # Test entry for Derm7pt
├── test_v2.py # Test entry for PH2
├── data.py # Data loading & preprocessing
├── promptCustom/
│ ├── context_guided_coop.py # Clinical-guided context optimization (core)
│ ├── prompt_tuning.py # Prompt learning module
│ ├── prompt_test.py # Prompt loading during testing
│ ├── clip.py # CLIP model loader
│ └── simple_tokenizer.py # Text tokenizer
├── train.sh # Training script
├── test.sh # Testing script
└── requirements.txt # Python dependencies

---

## 🧩 Model Features

### 1. Clinical-guided Context Optimization (CgCoOp)
- Uses 7-point checklist to construct fixed clinical prompts that guide learnable prompts
- Introduces cosine similarity loss to align text features with clinical features

### 2. Partial Image Encoder Fine-tuning (IFT)
- Only fine-tunes the last layer of visual encoder (`visual.ln_post`)
- Preserves general visual features while adapting to medical domain
- Reduces overfitting and improves generalization

---

## 🛠️ Environment Setup

**Verified Environment:**
- PyTorch 2.4.1 + CUDA 12.1
- NVIDIA A10 GPU

---

## 🚀 Training & Testing

### Training
```bash
bash train.sh

📊 Reproduction Results
On Derm7pt test set, this reproduction achieves:

Accuracy (ACC)	AUC	F1-score
82.53%	85.08%	81.19%
Best model saved at epoch 90, showing comparable performance to original paper.

🧪 Experiments
Main Experiment: main.py (full model)

Ablation Studies:

main_woIft.py: without image fine-tuning

main_wocg.py: without clinical guidance

main_woIft_cg.py: CoOp baseline only

Comparison: main_kgcoop.py (KgCoOp method)

📁 Model Files
After training, the following files are saved:

prompt_model_90.pt: text prompt model

Image_encoder_tuning90.pt: image encoder model

🙏 Acknowledgments
Thanks to the original authors from Fudan University for providing code and methodology.

This reproduction is based on the original paper for academic validation and further research.

📜 License
This project is for academic research only. Please follow the original paper and dataset licenses for data and code usage.