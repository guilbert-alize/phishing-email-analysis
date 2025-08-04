#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Setup for inline plotting (only needed in Jupyter)
# get_ipython().run_line_magic('matplotlib', 'inline')  # <- Remove if running as script

# Télécharger les ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('english')

# === PATH SETUP ===
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')
output_dir = os.path.join(base_dir, '..', 'python')

os.makedirs(output_dir, exist_ok=True)

# === Load Data ===
emails = pd.read_csv(os.path.join(data_dir, 'cleaned_emails.csv'))

# === Exploration initiale ===
print("Aperçu des données :")
print(emails.head())
print("\nInformations :")
print(emails.info())
print("\nRépartition des classes :")
print(emails['is_phishing'].value_counts())
print("\nStatistiques de text_length :")
print(emails['text_length'].describe())

# === Outliers inspection ===
print("\nEmails avec text_length > 10000 :")
print(emails[emails['text_length'] > 10000][['email_id', 'text_length', 'is_phishing', 'email_text']].head())

# === Visualisations ===
phishing_emails = emails[emails['is_phishing'] == 1]
no_phishing_emails = emails[emails['is_phishing'] == 0]

# Scatterplots
plt.figure(figsize=(12, 6))
sns.scatterplot(x='email_id', y='text_length', hue='is_phishing', data=phishing_emails, sizes=(20, 200))
plt.ylim(0, phishing_emails['text_length'].quantile(0.99))
plt.title("Longueur des emails (Phishing)")
plt.savefig(os.path.join(output_dir, 'text_length_scatterplot_phishing.png'))
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x='email_id', y='text_length', hue='is_phishing', data=no_phishing_emails, sizes=(20, 200))
plt.ylim(0, no_phishing_emails['text_length'].quantile(0.99))
plt.title("Longueur des emails (Non-Phishing)")
plt.savefig(os.path.join(output_dir, 'text_length_scatterplot_no_phishing.png'))
plt.show()

# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x='is_phishing', y='text_length', data=emails)
plt.ylim(0, emails['text_length'].quantile(0.99))
plt.title("Distribution de la longueur des emails par type")
plt.savefig(os.path.join(output_dir, 'text_length_boxplot.png'))
plt.show()

# === Détection des outliers ===
Q1 = np.percentile(emails['text_length'], 25)
Q3 = np.percentile(emails['text_length'], 75)
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
lower = max(0, Q1 - 1.5 * IQR)

print("Upper Bound:", upper, "Nombre d'outliers supérieurs:", (emails['text_length'] >= upper).sum())
print("Lower Bound:", lower, "Nombre d'outliers inférieurs:", (emails['text_length'] <= lower).sum())

# === WordClouds ===
phishing_words = " ".join(emails[emails['is_phishing'] == 1]['email_text'])
non_phishing_words = " ".join(emails[emails['is_phishing'] == 0]['email_text'])

phishing_wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(phishing_words)
non_phishing_wc = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(non_phishing_words)

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.imshow(phishing_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Phishing Emails (is_phishing = 1)', fontsize=16)

plt.subplot(1, 2, 2)
plt.imshow(non_phishing_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Non-Phishing Emails (is_phishing = 0)', fontsize=16)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'wordclouds.png'))
plt.show()

# === Export word frequencies ===
phishing_word_freq = pd.Series(phishing_words.split()).value_counts().head(50)
non_phishing_word_freq = pd.Series(non_phishing_words.split()).value_counts().head(50)

phishing_word_freq.to_csv(os.path.join(output_dir, 'phishing_word_freq.csv'), header=['frequency'])
non_phishing_word_freq.to_csv(os.path.join(output_dir, 'non_phishing_word_freq.csv'), header=['frequency'])

# === Vectorisation & Modélisation ===
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)
X = vectorizer.fit_transform(emails['email_text'].fillna(''))
y = emails['is_phishing']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\nPrécision du modèle :", accuracy)
print("\nRapport de classification :\n", class_report)

# === Confusion Matrix Plot ===
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Phishing', 'Phishing'], yticklabels=['Non-Phishing', 'Phishing'])
plt.title("Matrice de confusion - Détection des emails de phishing")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.show()

# === Coefficient Analysis ===
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]
coef_df = pd.DataFrame({'Mot': feature_names, 'Coefficient': coefficients})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

print("\nMots les plus associés aux emails de phishing (top 10) :")
print(coef_df.head(10))
print("\nMots les moins associés aux emails de phishing (top 10) :")
print(coef_df.tail(10))

# Barplots
plt.figure(figsize=(12, 6))
sns.barplot(x='Coefficient', y='Mot', data=coef_df.head(10), palette='Reds')
plt.title("Mots les plus prédictifs des emails de phishing")
plt.savefig(os.path.join(output_dir, 'top_phishing_words.png'))
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Coefficient', y='Mot', data=coef_df.tail(10), palette='Greens')
plt.title("Mots les moins prédictifs des emails de phishing")
plt.savefig(os.path.join(output_dir, 'top_non_phishing_words.png'))
plt.show()

# Exporter les coefficients pour Power BI
coef_df.to_csv(os.path.join(output_dir, 'coefficient_analysis.csv'), index=False)
