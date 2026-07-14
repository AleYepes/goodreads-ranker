# Goodreads Recommendations

Find books that match your tastes with this Goodreads crawler + recommendation pipeline.

## Getting Started

### 0. **Use Goodreads!**

<img src="assets/button.png" alt="alt text" width="300">

<br>The more books you rate (and the more users you befriend), the better your recommendations.

### 1. **Set Up**

#### A. Virtual environment

```
python3 -m venv venv
```

#### B. Required packages

```bash
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
nbstripout --install
nbdime config-git --enable
```

#### C. Ollama

```bash
brew install ollama
```

### 2. **Run the Pipeline**

```bash
venv/bin/python3 main.py run_pipeline
```

If you have a large library, or many friends, **this can take several hours** when run for the first time.

### 3. **Review the recommendations** (WIP\*)

Once the pipeline has run successfully, you can review your recommendations in `notebooks/explore_predictions.ipynb`

- Predictions are stored in `data/goodreads.db` under the `book_predictions` table.
