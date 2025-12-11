# Spine MRI Multimodal CLIP & Generation

This project implements a multimodal AI pipeline for **Lumbar Spine MRI analysis**, focusing on aligning 2.5D MRI sequences with radiology reports and generating summaries from images.

## 📌 Key Features
1.  **Text Summarization**: Extracts key findings from full radiology reports using **BioBART**.
2.  **2.5D Image Encoding**: Processes stacked MRI slices using **ConvNeXt**.
3.  **CLIP Training**: Aligns MRI images and text summaries via contrastive learning.
4.  **Image-to-Text Generation**: Generates diagnostic summaries directly from 2.5D MRI scans.

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python and PyTorch installed. Install the required dependencies:
```bash
pip install -r requirements.txt