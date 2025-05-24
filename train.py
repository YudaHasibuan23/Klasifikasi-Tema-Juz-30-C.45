import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv('data/juz30.csv')

# Encoding label tema
le = LabelEncoder()
df['tema_encoded'] = le.fit_transform(df['tema'])

# TF-IDF fitur dari teks latin
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['latin'])
y = df['tema_encoded']

# C4.5 model (Decision Tree with entropy)
model = DecisionTreeClassifier(criterion='entropy', max_depth=10)
model.fit(X, y)

# Simpan model
joblib.dump(model, 'model/c45_model.pkl')
joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')
joblib.dump(le, 'model/label_encoder.pkl')
