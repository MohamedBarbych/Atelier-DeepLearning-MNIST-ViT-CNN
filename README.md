# Lab-DeepLearning-MNIST-ViT-CNN

> Université Abdelmalek Essaâdi – FST Tanger  
> Master MBD – Deep Learning | Encadré par Pr. ELAACHAK LOTFI  
> Lab 2 – Vision par ordinateur avec PyTorch (CNN, VGG, ViT)

---

## 🎯 Objectif du TP

L’objectif principal de cet atelier est de se familiariser avec la bibliothèque **PyTorch** pour construire et entraîner différentes **architectures neuronales** appliquées à la vision par ordinateur. Nous avons exploré des réseaux classiques comme **CNN**, des modèles pré-entraînés (**VGG16, AlexNet**), ainsi que les **Transformers Visuels (ViT)**, sur le dataset **MNIST**.

---

## 📂 Contenu

### Partie 1 – Classification avec CNN

#### 🧠 Étapes réalisées :
1. **Création d’un modèle CNN personnalisé** (convolutions, maxpool, fully connected).
2. **Utilisation de GPU (RTX 2060)** via `torch.device("cuda")`.
3. **Entraînement et évaluation du modèle CNN** (Accuracy, F1-score, Loss, Training Time).
4. **Fine-tuning de modèles pré-entraînés** : `VGG16` et `AlexNet`.
5. **Comparaison des performances entre CNN, VGG16, AlexNet.**

#### 🛠️ Spécificités du CNN :
- Deux couches de convolution suivies de ReLU et MaxPool.
- Flatten → Fully Connected → Dropout → Classification.
- Optimiseur : Adam, Loss : CrossEntropy.

---

### Partie 2 – Vision Transformer (ViT)

#### 🔍 Étapes suivies :
- Implémentation d’un modèle **ViT from scratch**, inspiré de l'article [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929).
- Utilisation de la méthode `PatchEmbedding`, position embeddings, et multi-head self-attention.
- Application sur le dataset **MNIST** après adaptation des dimensions (résolution et canaux).
- Entraînement, évaluation, puis **comparaison avec les modèles CNN & VGG**.

---

## 📊 Résultats Comparatifs

| Modèle     | Accuracy (%) | F1-Score (%) | Temps d'entraînement |
|------------|--------------|--------------|------------------------|
| CNN        | 98.00        | 97.90        | ~1 min                 |
| VGG16      | 99.20        | 99.15        | ~2 min                 |
| AlexNet    | 98.50        | 98.40        | ~1.5 min               |
| Vision Transformer (ViT) | 97.10        | 96.90        | ~3 min                 |

📌 *Ces chiffres sont obtenus après 3 à 5 epochs d'entraînement sur GPU (RTX 2060).*

---

## 🧠 Analyse & Interprétation

- Le **modèle CNN personnalisé** fonctionne très bien sur MNIST, montrant la puissance des convolutions sur des données simples.
- Le **VGG16** fine-tuné dépasse clairement les performances du CNN classique, montrant l’intérêt du transfert d’apprentissage.
- Le **ViT**, bien que légèrement en dessous sur MNIST, reste compétitif même sans CNN, prouvant sa généralisation.
- Les **temps d'entraînement** montrent que les modèles Transformers sont plus gourmands que CNN/VGG pour ce dataset simple.

---

## 🧪 Enseignements & Conclusion

✅ Ce TP m’a permis de :
- Maîtriser les concepts de **convolutions, pooling, FC layers, dropout, etc.**
- Comprendre le fonctionnement de **modèles pré-entraînés** et comment les **adapter** à un nouveau jeu de données.
- Implémenter un **modèle Transformer visuel (ViT)** étape par étape.
- Comparer plusieurs architectures de vision par ordinateur de manière rigoureuse à l’aide de métriques (Accuracy, F1, Time).

---

## 🚀 Outils & Environnement

- 🔥 **PyTorch** 2.x
- 📈 **Sklearn** pour les métriques
- 🎮 **GPU** local : NVIDIA RTX 2060
- 🧪 Jupyter Notebook
- 📚 Dataset : [MNIST Digits Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

---

## 📁 Structure du Projet
📦 Atelier-DeepLearning-MNIST-ViT-CNN ┣ 📜 Atelier2_final.ipynb ← Notebook complet (exécutable) ┣ 📜 README.md ← Ce fichier ┗ 📂 data/ ← Téléchargé automatiquement par PyTorch

---

## ✅ À faire pour la validation

- [x] Implémenter CNN et fine-tuning VGG16/AlexNet
- [x] Implémenter ViT from scratch
- [x] Comparer tous les modèles
- [x] Synthèse finale dans le `README.md`
- [x] Héberger le projet sur GitHub

---

## 🔗 Références

- [Tutorial ViT Medium](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)
- [Paper ViT original](https://arxiv.org/abs/2010.11929)
- [Dataset MNIST Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

---

> *Fait avec passion ❤️ par [Mohamed BARBYCH]*

