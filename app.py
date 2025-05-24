from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load data dan model Algoritma C.45
df = pd.read_csv('data/juz30.csv')
model = joblib.load('model/c45_model.pkl')
tfidf = joblib.load('model/tfidf_vectorizer.pkl')
le = joblib.load('model/label_encoder.pkl')

# Prediksi tema untuk semua ayat (opsional)
X = tfidf.transform(df['latin'])
df['tema_prediksi'] = le.inverse_transform(model.predict(X))

@app.route('/')
def index():
    # Ambil tema asli dari dataset agar lengkap
    tema_list = sorted(df['tema'].dropna().unique()) 
    return render_template('index.html', tema_list=tema_list)

@app.route('/hasil', methods=['POST'])
def hasil():
    tema = request.form['tema']

    # SELALU ambil dari kolom asli untuk kelengkapan data
    hasil = df[df['tema'].str.lower() == tema.lower()]

    hasil = hasil[['surat', 'terjemahan', 'nama_surat', 'nama_surat_arab', 'no_surat']]
    
    return render_template('hasil.html', tema_dipilih=tema, hasil=hasil.to_dict(orient='records'))

if __name__ == '__main__':
import os
app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
