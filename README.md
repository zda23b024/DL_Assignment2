# DA6401 Assignment 2  
## Complete Visual Perception Pipeline on Oxford-IIIT Pet Dataset

---

##  Overview

This project implements a complete **visual perception pipeline** using the Oxford-IIIT Pet Dataset. The pipeline consists of three main tasks:

-   Image Classification (Pet Breed Classification)
-   Object Localization (Bounding Box Detection)
-   Semantic Segmentation (Pixel-level Pet Masking)

All components are built using a **VGG11-based encoder**, and experiments were conducted to analyze how architectural and training choices affect performance.

---

##  Tasks Implemented

### 🔹 Q2.1 – Batch Normalization Analysis
- Compared models **with and without BatchNorm**
- Observed:
  - Faster convergence with BN
  - Better validation performance
  - Stable activation distributions
- Insight: **BatchNorm improves optimization and generalization**

---

###  Q2.2 – Dropout Regularization
- Tested:
  - No dropout
  - Dropout (p = 0.2)
  - Dropout (p = 0.5)
- Observed:
  - No dropout → overfitting
  - p = 0.2 → best generalization
  - p = 0.5 → underfitting
- Insight: **Moderate dropout gives best performance**

---

###  Q2.3 – Transfer Learning Strategies
- Compared:
  - Frozen encoder
  - Partial fine-tuning
  - Full fine-tuning
- Results:
  - Frozen → lowest Dice (~0.77)
  - Partial → balanced (~0.79)
  - Full → best (~0.82)
- Insight: **Fine-tuning improves segmentation but increases cost**

---

###  Q2.4 – Feature Map Visualization
- Visualized:
  - First convolutional layer
  - Last convolutional layer
- Observed:
  - Early layers → edges & textures
  - Deep layers → semantic features (face, body)
- Insight: **CNNs learn hierarchical representations**

---

###  Q2.5 – Object Localization Analysis
- Evaluated bounding box predictions using:
  - Confidence scores
  - IoU (Intersection over Union)
- Observed:
  - High confidence (~0.7 avg)
  - Very low IoU (~0)
- Insight:  
  **Model detects object presence but struggles with precise localization**

---

###  Q2.6 – Segmentation Metrics
- Compared:
  - Pixel Accuracy
  - Dice Score
- Observed:
  - Pixel Accuracy high (background dominance)
  - Dice lower (focus on object overlap)
- Insight:  
  **Dice is a better metric for segmentation tasks**

---

###  Q2.7 – Final Pipeline Showcase
- Tested pipeline on **3 real-world internet images**
- Outputs:
  - Bounding boxes
  - Segmentation masks
  - Cropped images
  - Predicted breed + confidence
- Observed:
  - High confidence for familiar images
  - Low confidence for difficult cases
- Insight:  
  **Pipeline works but struggles with domain shift**

---

###  Q2.8 – Meta Analysis
- Combined insights from all experiments:
  - BatchNorm → stabilizes training
  - Dropout → controls overfitting
  - Transfer Learning → improves adaptation
  - Dice > Pixel Accuracy for segmentation
- Insight:  
  **Design choices strongly impact performance of multi-task systems**

---

##  Key Learnings

- ✔ Proper normalization improves convergence  
- ✔ Regularization must be balanced  
- ✔ Fine-tuning improves task-specific performance  
- ✔ Evaluation metrics must match the task  
- ✔ Multi-task learning introduces trade-offs  

---

##  W&B Report

 Full detailed report (plots, analysis, visualizations):

**[View W&B Report](https://wandb.ai/zda23m016-iit-madras-zanzibar/da6401_assignment2/reports/DA6401-Assignment-2-Complete-Visual-Perception-Pipeline-on-Oxford-IIIT-Pet-Dataset--VmlldzoxNjQyMDIxMA?accessToken=ejnjvpr3flreyoay2oxf8vujj912pfxogyg7l50gockh8qnveow6uhx5tiiba6zq)**

---

##  Tech Stack

- Python
- PyTorch
- Weights & Biases (W&B)
- NumPy / Matplotlib

---


---

##  Conclusion

This project demonstrates how a single backbone (VGG11) can be extended into a full perception pipeline. While the system performs well on standard data, challenges such as **localization accuracy and generalization to real-world images** remain important areas for improvement.

---

⭐ **Overall: A complete end-to-end visual perception system with detailed experimental analysis.**
