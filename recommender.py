# 1. Imports + load data
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pf = pd.read_csv("raw_skills.csv")

# 2. Build job role vectors
vectorizer = TfidfVectorizer()
job_vectors = vectorizer.fit_transform(pf["skills"])

# 3. Get user input
skill1 = input("Type skill 1: ")
skill2 = input("Type skill 2: ")
skill3 = input("Type skill 3: ")

# 4. Clean spaces
skill1 = skill1.replace(" ", "")
skill2 = skill2.replace(" ", "")
skill3 = skill3.replace(" ", "")

# 5. Combine
user_skills = skill1 + " " + skill2 + " " + skill3

# 6. Vectorize user input
user_vector = vectorizer.transform([user_skills])

# 7. Cosine similarity
similarities = cosine_similarity(user_vector, job_vectors)

# 8. Top 3
top_3_indices = similarities[0].argsort()[-3:][::-1]

print("\nTop 3 recommended roles:")
print("-" * 25)

rank = 1
for i in top_3_indices:
    role_name = pf["role"][i]
    match_percent = similarities[0][i] * 100
    print(f"{rank}. {role_name} - {match_percent:.1f}% match")
    rank += 1