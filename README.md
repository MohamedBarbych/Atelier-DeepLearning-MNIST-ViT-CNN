# Lab-DeepLearning-MNIST-ViT-CNN

> Abdelmalek Essaâdi University – FST Tangier  
> Master MBD – Deep Learning | Instructor: Prof. ELAACHAK LOTFI  
> Lab 2 – Computer Vision with PyTorch (CNN, VGG, ViT)

---

## 🎯 Lab Objective

The main goal of this lab is to get familiar with the **PyTorch** library by building and training various **neural network architectures** for computer vision tasks. We explore classic architectures like **CNN**, fine-tune **pretrained models** (VGG16, AlexNet), and implement a **Vision Transformer (ViT)** from scratch using the **MNIST dataset**.

---

## 📂 Lab Content

### Part 1 – CNN Classifier

#### 🧠 Completed Steps:
1. Built a **custom CNN model** using PyTorch (convolution, max-pooling, fully connected).
2. Trained the model on **GPU (RTX 2060)** using `torch.device("cuda")`.
3. Evaluated the CNN model using **accuracy, F1-score, loss, and training time**.
4. Fine-tuned **pretrained models**: `VGG16` and `AlexNet` on MNIST.
5. **Compared performances** between CNN, VGG16, and AlexNet.

#### 🛠️ CNN Architecture Details:
- Two convolutional layers + ReLU + MaxPooling.
- Followed by Flatten → Fully Connected → Dropout → Output layer.
- Optimizer: Adam, Loss Function: CrossEntropyLoss.

---

### Part 2 – Vision Transformer (ViT)

#### 🔍 Process:
- Implemented a **ViT model from scratch**, inspired by [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929).
- Used `PatchEmbedding`, learnable positional embeddings, and multi-head self-attention.
- Adapted the input dimensions (resolution and channels) to work with MNIST.
- Trained and evaluated the model, then **compared its results** to CNN and VGG models.

---

## 📊 Comparative Results

| Model               | Accuracy (%) | F1-Score (%) | Training Time |
|---------------------|--------------|--------------|----------------|
| CNN                 | 98.00        | 97.90        | ~1 min         |
| VGG16               | 99.20        | 99.15        | ~2 min         |
| AlexNet             | 98.50        | 98.40        | ~1.5 min       |
| Vision Transformer  | 97.10        | 96.90        | ~3 min         |

📌 *These results were obtained after 3 to 5 training epochs using a GPU (NVIDIA RTX 2060).*

---

## 🧠 Analysis & Interpretation

- The **custom CNN model** performs very well on MNIST, proving the strength of convolutions on digit recognition.
- The **VGG16 fine-tuned model** outperforms the basic CNN, demonstrating the effectiveness of **transfer learning**.
- The **Vision Transformer (ViT)**, although slightly behind CNN on MNIST, achieves solid performance without using convolutions, confirming its generalization capabilities.
- **Training time** analysis shows that transformer-based models are heavier and more computationally intensive than CNN/VGG on simple datasets.

---

## 🧪 Key Learnings & Conclusion

✅ Through this lab, I was able to:

- Master essential concepts like **convolution, pooling, dropout, fully connected layers**.
- Understand how to **use and fine-tune pretrained models** on new datasets.
- Implement a **Vision Transformer (ViT)** architecture step-by-step from scratch.
- Compare multiple deep learning models for computer vision using standard evaluation metrics (Accuracy, F1-Score, Training Time).

---

## 🚀 Tools & Environment

- 🔥 **PyTorch** 2.x
- 📈 **Sklearn** for metrics
- 🎮 **Local GPU**: NVIDIA RTX 2060
- 🧪 Jupyter Notebook
- 📚 Dataset: [MNIST Digits Dataset – Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

---

## 📁 Project Structure

📦 Lab-DeepLearning-MNIST-ViT-CNN ┣ 📜 Atelier2_final.ipynb ← Fully functional notebook ┣ 📜 README.md ← This file ┗ 📂 data/ ← Automatically created by PyTorch on first run


---

## ✅ Checklist for Completion

- [x] Implement custom CNN model
- [x] Fine-tune VGG16 and AlexNet on MNIST
- [x] Implement Vision Transformer from scratch
- [x] Compare all models using accuracy, F1-score, and training time
- [x] Write the final synthesis in this `README.md`
- [x] Upload the project to GitHub

---

## 🔗 References

- [ViT Tutorial – Medium](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)
- [ViT Original Paper – arXiv](https://arxiv.org/abs/2010.11929)
- [MNIST Dataset – Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

---

> *Crafted with passion ❤️ by Mohamed BARBYCH*
