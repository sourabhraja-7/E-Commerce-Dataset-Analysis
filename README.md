# E-Commerce Dataset Analysis (PySpark + ML)

This project analyzes a large-scale e-commerce clickstream dataset (2019-Oct) using **PySpark** for scalable preprocessing and analytics, followed by **EDA** and **machine learning** to study purchase behavior and segment users.

## What’s inside the notebook

### 1) PySpark Setup & Data Loading
- Runs on **Google Colab + Spark**
- Loads raw dataset from Google Drive (`2019-Oct.csv`)
- Casts types (timestamp, price as double) and selects relevant columns

### 2) Data Cleaning (Spark)
- Fills missing **category_code** and **brand** using the most frequent value per **product_id**
- Drops rows with unknown category/brand after imputation
- Drops non-essential columns (e.g., `user_session`, `category_id` when present)
- Adds time-based columns: `event_date`, `day_of_week`, `hour_of_day`
- Extracts `main_category` from `category_code`
- Writes cleaned dataset to Parquet for reuse  
  Final cleaned size: **26,611,226 rows**

### 3) Exploratory Data Analysis (EDA)
- Event type distribution (view/cart/purchase)
- Daily interaction trends
- Top categories and top brands by interaction volume
- Funnel insight: large drop-off from views → cart → purchase

## Machine Learning Tasks

### Problem 1 — Purchase Prediction (Spark ML)
- Creates binary label: `purchase = 1`, else `0`
- Builds Spark ML pipeline: **StringIndexer → OneHotEncoder → VectorAssembler → LogisticRegression**
- Evaluates using metrics like accuracy and AUC (includes ROC/PR analysis plots)

### Problem 2 — User Segmentation (KMeans Clustering)
- Aggregates per-user behavioral features:
  `num_view, num_cart, num_purchase, num_products, num_categories, avg_price`
- Standardizes features and trains **KMeans**
- Picks best k using **Silhouette Score**
  Best k: **3** (Silhouette ≈ **0.721**)
- Visualizes clusters using **PCA scatter plot** and **cluster center heatmap**

### Problem 3 — Price-Based Purchase Prediction
- Trains Logistic Regression using **price + (brand, category_code)** encodings
- Reports classification metrics (Accuracy/F1/AUC) and plots (confusion matrix, precision-recall)

## Tech Stack
- **PySpark** (data cleaning + Spark ML pipelines)
- **Pandas / NumPy** (analysis helpers)
- **Matplotlib / Seaborn** (plots)
- **Scikit-learn** (PCA, classification report, confusion matrix, ROC/PR)


