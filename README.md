# Tech Stack Recommender

A skill-based job role recommender that takes a user's 3 skills as input and recommends the top 3 best-matching job roles, using **TF-IDF vectorization** and **cosine similarity** — not simple keyword counting.

## How It Works

Simple keyword counting can create ties between roles — two different roles can both match 2 out of 3 user skills, with no way to tell which is the better fit. This happens because counting treats every skill as equally important, even though some skills (like "MachineLearning") are far more specific and rare than common skills (like "Python").

**TF-IDF (Term Frequency–Inverse Document Frequency)** solves this by weighting each skill based on how rare or specific it is across all job roles. Common skills that appear in many roles get lower weight, while rare, specialized skills get higher weight — allowing the system to break ties that simple counting can't.

Each job role's skill list is converted into a TF-IDF vector. The user's 3 input skills are converted into a vector using the same vocabulary. **Cosine similarity** then measures the angle between the user's vector and each job role's vector, producing a score between 0 (no alignment) and 1 (perfect alignment). The 3 roles with the highest similarity scores are recommended.

## Tech Stack

- Python
- pandas
- scikit-learn (`TfidfVectorizer`, `cosine_similarity`)

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Run the script:
python recommender.py

3. Enter 3 skills when prompted (multi-word skills like "Machine Learning" are supported — spaces are automatically stripped to match the internal format).

## Example
Type skill 1: machine learning
Type skill 2: python
Type skill 3: statistics
Top 3 recommended roles:
Data Scientist - 0.56
Machine Learning Engineer - 0.503
Data Analyst - 0.166

## Dataset

`raw_skills.csv` contains 8 job roles (Data Scientist, DevOps Engineer, Backend Developer, Cloud Architect, Frontend Developer, Data Analyst, Machine Learning Engineer, Cybersecurity Analyst), each mapped to a space-separated list of associated skills.

## Notes

- Multi-word skills are stored as single joined tokens (e.g. `MachineLearning`) to keep them as one unit in the vocabulary. User input is automatically stripped of spaces to match this format.
- Matching is not exact-spelling-dependent for casing (TF-IDF lowercases automatically), but does require the skill to closely match a vocabulary word (e.g. "statistic" vs "Statistics" may not match exactly).