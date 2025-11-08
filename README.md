# CLIP-Based Melanoma Diagnosis (Reproduction)

🔬 This project is a reproduction of the paper **"A Melanoma Diagnosis Method Based on Large-Scale Vision-Language Models"** published in *Acta Anatomica Sinica* (2025). The original work proposes a CLIP-based framework enhanced with clinical prompt guidance and partial image encoder fine-tuning for melanoma classification from dermoscopic images.

---

## 📁 Project Structure
### 1. The code floder
- It contains five training files corresponding to different comparison or ablation experiments (see Section V. Comparison and Ablation Experiment Description in this document for details). Among them, the main.py file is the training file for the main experiment.
- The code file contains two test files: test.py and test_v2.py. test.py is the inference test file for the Derm7pt dataset, while test_v2.py is the inference test file for the Ph2 dataset.
- The data.py is the data construction file.

### 2. The promptCustom floder 
- The promptCustom/context_guided_coop.py file in the promptCustom folder is the file used to construct prompt learning. This file modifies the CoOp framework by adding clinical knowledge guidance. 
- The promptCuston/prompt_tuning_v2.py is the file used to load prompts during testing, following the same procedure as CoOp. The remaining files are publicly available code files from others' replications of CoOp/KgCoOp. 

### 3. The requirement file 
- It contains all the packages required for the runtime environment.

### 4. The output folder
- These documents comprehensively document the entire reproducible process from environment setup and model training to final testing, providing full support for the project's reproducibility.
- final_test_results (1).txt - Final Test Results 
It contains performance metrics of the model on the test set and records key metrics such as accuracy, AUC, and F1-score
- training_summary (1).txt - Training Summary
It records training completion status and key parameters including best epoch (90th iteration) and corresponding accuracy (82.53%)
It saves training environment and model path information
- validation_report (1).txt - Environment Validation Report
It verify model file integrity and validate PyTorch and CUDA environments recording GPU configuration and availability
- gpu_info (1).txt - GPU Information
Detailed specifications of the NVIDIA A10 GPU I rented on Alibaba Cloud, it includes driver version and CUDA version information
and verifies training hardware environment

- model_weights/logs - Model weights and logs directory
It stores log records during the training process and contains training visualization data such as loss curve and accuracy changes

- Model weight file (not uploaded yet due to large file size)
Includes the trained text prompt model (prompt_model_90.pt)
Includes the fine-tuned image encoder (Image_encoder_tuning90.pt)
Plans to upload via Git LFS or other large file storage solutions

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