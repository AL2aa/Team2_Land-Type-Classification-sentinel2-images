import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------------------
# Confusion Matrix
# -------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10,8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    ax.set_title("Confusion Matrix")
    return fig

# -------------------------------
# Sample predictions grid
# -------------------------------
def plot_sample_predictions(dataset, y_true, y_pred, class_names, n_samples=12, batch_size=32):
    fig, axes = plt.subplots(3, 4, figsize=(15,10))
    axes = axes.flatten()
    
    dataset_list = list(dataset)
    indices = random.sample(range(len(y_true)), n_samples)
    
    for i, idx in enumerate(indices):
        batch_idx = idx // batch_size
        img_idx = idx % batch_size
        img = dataset_list[batch_idx][0].numpy()[img_idx]
        
        axes[i].imshow(img.astype('uint8'))
        axes[i].axis('off')
        
        true_class = class_names[y_true[idx]]
        pred_class = class_names[y_pred[idx]]
        axes[i].set_title(f"T:{true_class}\nP:{pred_class}",
                          color='green' if true_class==pred_class else 'red', fontsize=10)
    
    plt.tight_layout()
    return fig

# -------------------------------
# Prediction probabilities bar chart
# -------------------------------
def plot_prediction_probs(pred_probs, class_names, true_class=None, pred_class=None):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(class_names, pred_probs, color='skyblue')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    
    title_text = "Prediction Probabilities"
    if true_class and pred_class:
        title_text += f" (True: {true_class}, Pred: {pred_class})"
    ax.set_title(title_text)
    return fig

# -------------------------------
# Random single image with prediction
# -------------------------------
def plot_random_image(dataset, y_true, y_pred, class_names, batch_size=32):
    random_idx = random.randint(0, len(y_true)-1)
    batch_idx = random_idx // batch_size
    img_idx = random_idx % batch_size
    
    img = list(dataset)[batch_idx][0].numpy()[img_idx]
    true_class = class_names[y_true[random_idx]]
    pred_class = class_names[y_pred[random_idx]]
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img.astype('uint8'))
    ax.axis('off')
    ax.set_title(f"True: {true_class}\nPred: {pred_class}",
                 color='green' if true_class==pred_class else 'red')
    
    return fig, true_class, pred_class