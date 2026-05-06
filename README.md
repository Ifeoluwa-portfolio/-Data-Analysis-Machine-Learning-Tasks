# -Data-Analysis-Machine-Learning-Tasks
### Python | Pandas | Scikit-learn | NLTK | TextBlob | Matplotlib | Seaborn

# OVERVIEW
This project covers a series of data analysis and machine learning tasks performed using Python in Jupyter Notebook (VS Code). The tasks involve data cleaning, visualization, time series analysis, classification modeling, clustering, and natural language processing (NLP) using real-world datasets.
## 🗂️ Table of Contents

Task 1 - Data Cleaning & Preprocessing
Task 2 - Basic Data Visualization
Task 3 - Time Series Analysis
Task 4 - Clustering Analysis (KMeans)
Task 5 - Predictive Modeling (Classification)
Task 6 - Sentiment Analysis (NLP)
Tools & Libraries
How to Run

## ✅ Task 1 - Data Cleaning & Preprocessing
**Dataset:** Iris Dataset (iris.csv)

### What Was Done:
- Loaded the Iris dataset using Pandas
- Checked and confirmed **zero missing values** across all 5 columns
- Identified and removed **3 duplicate rows** (150 → 147 rows)
- Checked and verified correct **data types** for all columns
- Standardized the **species** categorical column (setosa, versicolor, virginica)
- Saved the cleaned dataset as `iris_cleaned.csv`

### Key Results:
| Check | Result |
|-------|--------|
| Total Rows | 150 |
| Missing Values | 0 ✅ |
| Duplicate Rows Found | 3 |
| Rows After Cleaning | 147 |
| Species Categories | setosa, versicolor, virginica |

### Code Snippet:
```python
import pandas as pd

df = pd.read_csv(r'C:\Users\hp\Documents\iris.csv')

# Check missing values
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Standardize species column
df['species'] = df['species'].str.lower().str.strip()

# Save cleaned dataset
df.to_csv('iris_cleaned.csv', index=False)
```

---

## ✅ Task 2 - Basic Data Visualization
**Dataset:** Iris Dataset (iris.csv)
### What Was Done:
- Created **bar plots** showing species distribution
- Created **line charts** showing average measurements per species
- Created **scatter plots** for sepal and petal relationships
- Created **histograms** for feature distributions
- Created **boxplots** for each feature by species
- Generated **heatmaps** showing feature correlations
- Generated **pairplots** showing all feature relationships
- All plots exported as PNG images

### Plots Generated:
| Plot | File |
|------|------|
| Bar Plot | bar_plot.png |
| Line Chart | line_chart.png |
| Scatter Plot | scatter_plot.png |
| Histogram | histogram.png |
| Boxplot | boxplot.png |
| Heatmap | heatmap.png |
| Pairplot | pairplot.png |
| All Plots Combined | iris_all_plots.png |

### Code Snippet:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Pairplot
sns.pairplot(df, hue='species', palette='Set1')
plt.savefig('pairplot.png', dpi=300)
plt.show()
```
---

## ✅ Task 3 - Time Series Analysis
**Dataset:** Stock Prices Data Set (Stock Prices Data Set.csv)

### What Was Done:
- Loaded stock prices dataset with 497,472 rows and 7 columns
- Filtered for **Apple (AAPL)** stock data
- Fixed and set **date column** as index
- Plotted the **time series** of AAPL close prices (2014–2024)
- Calculated and plotted **Moving Averages** (7-day, 30-day, 90-day)
- **Decomposed** the time series into Trend, Seasonality, and Residuals
- Performed **ADF Stationarity Test** to check if data is stationary
- Applied **differencing** to make data stationary
- Analyzed **yearly and monthly patterns**

### Key Results:
| Metric | Value |
|--------|-------|
| Total Symbols | Multiple (filtered AAPL) |
| Date Range | 2014 - 2024 |
| Mean Close Price | ~$127 |
| ADF Test Result | Not Stationary (p > 0.05) |
| After Differencing | Stationary ✅ |

### Plots Generated:
- `aapl_time_series.png`
- `aapl_moving_average.png`
- `aapl_decomposition.png`
- `aapl_patterns.png`

### Code Snippet:
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series
decomposition = seasonal_decompose(
    df_aapl['close'],
    model='multiplicative',
    period=30
)

# Plot components
decomposition.plot()
plt.savefig('aapl_decomposition.png', dpi=300)
```

---

## ✅ Task 4 - Clustering Analysis (KMeans)
**Dataset:** Iris Dataset (iris.csv)

### What Was Done:
- Selected all 4 features for clustering
- Applied **StandardScaler** to normalize features
- Used the **Elbow Method** to find optimal number of clusters (K=3)
- Calculated **Silhouette Scores** for K=2 to 10
- Trained **KMeans model** with K=3 clusters
- Visualized clusters using scatter plots (sepal and petal features)
- Created **Cluster Centers Heatmap**
- Evaluated model using **Silhouette Score** and **Inertia**

### Key Results:
| Metric | Value |
|--------|-------|
| Optimal Clusters (K) | 3 |
| Silhouette Score | ~0.55 |
| Cluster Sizes | 62 / 50 / 38 |

### Plots Generated:
- `elbow_curve.png`
- `cluster_scatter.png`
- `silhouette_plot.png`
- `cluster_heatmap.png`
- `kmeans_complete.png`

### Code Snippet:
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
df['Cluster'] = kmeans.labels_
```

---

## ✅ Task 5 - Predictive Modeling (Classification)
**Dataset:** Churn Dataset (churn.csv)

### What Was Done:
- Loaded churn dataset with 667 rows and 20 columns
- Handled **categorical variables** using LabelEncoder
- Applied **StandardScaler** for feature scaling
- Split data into **80% training / 20% testing**
- Trained **3 classification models**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Evaluated all models using **Accuracy, Precision, Recall, F1-Score**
- Plotted **Confusion Matrices** for all models
- Performed **Hyperparameter Tuning** using GridSearchCV
- Identified **Top 10 Most Important Features** for churn prediction

### Key Results:
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | ~81% | ~0.60 |
| Decision Tree | ~78% | ~0.56 |
| Random Forest | ~82% | ~0.63 |
| Tuned Random Forest | ~85% | ~0.67 |

### Plots Generated:
- `churn_distribution.png`
- `confusion_matrix.png`
- `model_comparison.png`
- `feature_importance.png`

### Code Snippet:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='f1'
)
grid_search.fit(X_train, y_train)
```

---

## ✅ Task 6 - Sentiment Analysis (NLP)
**Dataset:** Sentiment Dataset (Sentiment_dataset.csv)

### What Was Done:
- Loaded sentiment dataset with **732 rows** and 15 columns
- Cleaned and stripped whitespace from sentiment labels
- Mapped **279 unique sentiments** into 3 main categories:
  - Positive, Negative, Neutral
- **Preprocessed text data**:
  - Converted to lowercase
  - Removed punctuation and numbers
  - Tokenized using NLTK
  - Removed stopwords
  - Applied Lemmatization
- Performed **TextBlob sentiment analysis**
- Calculated **Polarity** and **Subjectivity** scores
- Generated **Word Clouds** for each sentiment category
- Visualized **Top 10 words** per sentiment
- Compared original labels vs TextBlob predictions

### Key Results:
| Metric | Value |
|--------|-------|
| Total Texts | 732 |
| Missing Values | 0 ✅ |
| Avg Polarity | 0.0957 (Slightly Positive) |
| Avg Subjectivity | 0.3407 (Mostly Objective) |
| Positive Texts | 25.8% |
| Neutral Texts | 65.8% |
| Negative Texts | 8.3% |

### Plots Generated:
- `sentiment_distribution.png`
- `sentiment_comparison.png`
- `wordcloud.png`
- `word_frequency.png`
- `polarity_subjectivity.png`

### Code Snippet:
```python
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df['Cleaned_Text'] = df['Text'].apply(preprocess_text)
df['Polarity'] = df['Cleaned_Text'].apply(
    lambda x: TextBlob(x).sentiment.polarity
)
```

---

## 🛠️ Tools & Libraries
```
Python          3.11
Pandas          Data loading and manipulation
NumPy           Numerical computations
Matplotlib      Data visualization
Seaborn         Statistical visualization
Scikit-learn    Machine learning models
Statsmodels     Time series decomposition
NLTK            Natural language processing
TextBlob        Sentiment analysis
WordCloud       Word cloud visualization
Jupyter         Notebook environment (VS Code)
```

---

## 📁 Datasets Used
| Dataset | Rows | Columns | Task |
|---------|------|---------|------|
| iris.csv | 150 | 5 | Data Cleaning, Visualization, KMeans |
| Stock Prices Data Set.csv | 497,472 | 7 | Time Series Analysis |
| churn.csv | 667 | 20 | Classification Modeling |
| Sentiment_dataset.csv | 732 | 15 | Sentiment Analysis (NLP) |

---

## ▶️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/data-analysis-tasks.git
cd data-analysis-tasks
```

### 2. Install Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels nltk textblob wordcloud
```

### 3. Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 4. Open Jupyter Notebook
```bash
jupyter notebook data_analysis_first_task.ipynb
```

### 5. Run All Cells
- Click **Kernel** → **Restart & Run All**

  ---

## 👤 Author
**Oguntula Ifeoluwa Zaccheus**
Data Analyst | Python | SQL | Power BI | Excel
