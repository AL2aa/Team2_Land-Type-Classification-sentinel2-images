Dataset: <a href="https://github.com/phelber/EuroSAT" target="_blank">EuroSAT</a>

# Land Type Classification App 🌍

This project is a Streamlit web application that uses a trained EfficientNetB0 deep-learning model to classify satellite images from the EuroSAT dataset into 10 land-cover classes.

---

## Requirements

- Python 3.10 or 3.11
- pip
- Virtual environment (recommended)

---

## Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/AL2aa/Team2_Land-Type-Classification-sentinel2-images
cd Team2_Land-Type-Classification-sentinel2-images
```

---

### 2. Create a virtual environment

**Windows**

```
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

Install required libraries inside the virtual environment:

```
pip install -r app/requirements.txt
```

---

### 4. Check required assets

Before running the app, ensure that:

- The trained model exists at:

  ```
  models/eurosat_effb0_best.keras
  ```

- The background image exists at:

  ```
  app/assets/background.jpg
  ```

- The visualization module exists at:

  ```
  app/visualization.py
  ```

---

## Running the App

From the project root:

```
streamlit run app/app.py
```

Streamlit will print a local URL, open that link in your browser to access the app.

---

## How to Use

1. Open the Streamlit app in the browser.
2. Upload a satellite image (`.jpg`, `.jpeg`, or `.png`).
3. The app:

   - Displays the selected image.
   - Resizes and preprocesses it to 128×128.
   - Runs inference using the trained model.
   - Displays the predicted land-type class.
   - Shows a probability bar chart for all 10 classes.

---

## Model Information

- Architecture: EfficientNetB0
- Dataset: EuroSAT
- Number of classes: 10

```
AnnualCrop
Forest
HerbaceousVegetation
Highway
Industrial
Pasture
PermanentCrop
Residential
River
SeaLake
```

---

## Stopping the App

Press `Ctrl + C` in the terminal to stop Streamlit.

---

## Deactivating the Virtual Environment

```
deactivate
```
