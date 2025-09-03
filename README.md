# CS771 Mini-Project 1: Binary Classification on Three Feature Representations

**Team: Parameter Hunters**

---

## 🎯 Problem Statement  
Train accurate binary classifiers under a **10 000-parameter** limit using three feature paradigms—Emoticons, Deep Features, and Text Sequences—from the same raw dataset. Evaluate each representation individually with varying training-data percentages and assess whether a combined model further boosts validation accuracy.

---

## 📂 Dataset Structure  
- **Training / Validation / Test Sizes**: 7 080 / 489 / (hidden) examples  
- **Emoticons as Features** (`.csv`): 13 categorical emoticon columns → label  
- **Deep Features** (`.npz`): 13 × 786 real embeddings → label  
- **Text Sequences** (`.csv`): 50-digit string → label  
- Validation labels provided; test labels withheld.

---

## 🚀 Approach  

### Task 1: Individual Dataset Modeling  
1. **Emoticons**  
   - One-hot positional encoding + permutation importance to prune 7 least informative emojis  
   - Model: Logistic Regression (621 features + bias = 622 params)  
2. **Deep Features**  
   - Flatten 13×786 → 9 984-dim vector; L1 regularization identifies top rows  
   - Model: Logistic Regression on flattened features (≈ 9 985 params)  
3. **Text Sequences**  
   - Experiments:  
     - Categorical Naive Bayes on grouped digits (K=3)  
     - LSTM on integer-encoded sequences  
     - Custom token extraction + Embedding → FFNN (9 029 params)  

For each, trained on {20%, 40%, 60%, 80%, 100%} data and plotted training-size vs. validation accuracy.

### Task 2: Combined Dataset Modeling  
1. **Ensemble Voting** (majority & weighted by individual accuracies)  
2. **Confidence-Based Selection** (choose per-sample prediction from most confident model)  
3. **Feature Concatenation** (shape 10 623) + classifiers: Logistic Regression, KNN, FFNN, GB, SVM, Random Forest  

**Best Combined Model**: Random Forest Classifier (validation accuracy up to **99.39%**).

---

## 📊 Key Results  

| Dataset               | Best Model                       | Params  | 20% Acc. | 100% Acc. |
|-----------------------|----------------------------------|---------|----------|-----------|
| Emoticons             | Logistic Regression              | 622     | 88.34%   | 97.14%    |
| Deep Features         | Logistic Regression (flattened)  | 9 985   | 55.25%   | 98.77%    |
| Text Sequences        | FFNN (custom tokens)             | 9 029   | 91.00%   | 98.00%    |
| **Combined Features** | Random Forest Classifier         | –       | 97.14%   | 99.39%    |

- **Ensemble Voting**: 98.10% @100%  
- **Confidence-Based**: 98.50% @100%  
- **Feature Concatenation + RF**: 99.39% @100%  

---

## 🛠️ Implementation  

- **Language & Frameworks**: Python, NumPy, pandas, scikit-learn, TensorFlow/Keras  
- **Entry Point**:  
  - `read_data.py`: Data ingestion  
  - `your_group_no.py`: Full pipeline → generates  
    - `pred_emoticon.txt`  
    - `pred_deepfeat.txt`  
    - `pred_textseq.txt`  
    - `pred_combined.txt`  
- **Reproducibility**: Fixed random seeds; best checkpoints saved; environment and dependency details in `requirements.txt`.

---

## 🎓 Contributions  

1. **Emoji Pruning**: Permutation importance for feature reduction  
2. **Deep Embed. Reduction**: L1-regularized row selection  
3. **Token-Driven Text Modeling**: Custom substrings-based tokenization  
4. **Unified Representation**: Feature concatenation + RF for superior accuracy  

---

## 🔗 Resources  

- **Data & Code**: https://tinyurl.com/cs771-autumn24-mp-1-data  
- **Course**: CS771 – Introduction to Machine Learning, IIT Kanpur  

---

## 📜 License  
This project is released under the MIT License. See [LICENSE](LICENSE) for details.  
