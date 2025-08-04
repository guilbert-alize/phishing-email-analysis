#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import sqlite3
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Télécharger les ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# In[4]:


# Charger les données
emails = pd.read_csv('C:/Users/guilb/Documents/CV/skills/Phishing_Email_Dataset/data/cleaned_emails.csv')


# In[5]:


# Exploration initiale
print("Aperçu des données :")
print(emails.head())
print("\nInformations :")
print(emails.info())
print("\nRépartition des classes :")
print(emails['is_phishing'].value_counts())  # 11322 phishing, 7328 non-phishing
print("\nStatistiques de text_length :")
print(emails['text_length'].describe())


# In[8]:


# Identifier les outliers extrêmes
print("\nEmails avec text_length > 10000 :")
print(emails[emails['text_length'] > 10000][['email_id', 'text_length', 'is_phishing', 'email_text']].head())

#On peut voir que les outliers ne permettent pas d'identifier les mails spams par arpport aux mails non spam. 


# In[7]:


# Visualisations : Scatterplots et Boxplot
phishing_emails = emails[emails['is_phishing'] == 1]
no_phishing_emails = emails[emails['is_phishing'] == 0]


# In[13]:


# Scatterplots avec la longueur des emais spams
plt.figure(figsize=(12, 6))
sns.scatterplot(x='email_id', y='text_length', hue='is_phishing', data=phishing_emails, sizes=(20, 200))
plt.ylim(0, phishing_emails['text_length'].quantile(0.99))  # Limite au 99e centile
plt.title("Longueur des emails (Phishing)")
plt.savefig('C:/Users/guilb/Documents/CV/skills/Phishing_Email_Dataset/python/text_length_scatterplot_phishing.png')
plt.show()


# In[14]:


# Scatterplots avec la longueur des emais non-spams
plt.figure(figsize=(12, 6))
sns.scatterplot(x='email_id', y='text_length', hue='is_phishing', data=no_phishing_emails, sizes=(20, 200))
plt.ylim(0, no_phishing_emails['text_length'].quantile(0.99))
plt.title("Longueur des emails (Non-Phishing)")
plt.savefig('C:/Users/guilb/Documents/CV/skills/Phishing_Email_Dataset/python/text_length_scatterplot_no_phishing.png')
plt.show()


# In[15]:


# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x='is_phishing', y='text_length', data=emails)
plt.ylim(0, emails['text_length'].quantile(0.99))
plt.title("Distribution de la longueur des emails par type")
plt.savefig('C:/Users/guilb/Documents/CV/skills/Phishing_Email_Dataset/python/text_length_boxplot.png')
plt.show()

# On s'aperçoit que la distribution des mais spams est beaucoup plus importante avec des longeurs très petites ou très grandes


# In[16]:


# Détection des outliers avec IQR
Q1 = np.percentile(emails['text_length'], 25, method='midpoint')
Q3 = np.percentile(emails['text_length'], 75, method='midpoint')
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
lower = max(0, Q1 - 1.5 * IQR)  # Pas de longueurs négatives

upper_array = emails['text_length'] >= upper
print("Upper Bound:", upper, "Nombre d'outliers supérieurs:", upper_array.sum())
lower_array = emails['text_length'] <= lower
print("Lower Bound:", lower, "Nombre d'outliers inférieurs:", lower_array.sum())


# In[20]:


phishing_texts = emails[emails['is_phishing'] == 1]['email_text']
non_phishing_texts = emails[emails['is_phishing'] == 0]['email_text']
phishing_words = " ".join(phishing_texts)
non_phishing_words = " ".join(non_phishing_texts)


# In[21]:


# Create WordCloud objects
phishing_wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(phishing_words)
non_phishing_wc = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(non_phishing_words)

# Plot
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
plt.show()

# On s'aperçoit que les mails spams utilisent de nombreux caratères spéciaux, ainsu=i qu'un langage accrocheur : free, need, order, will, now, make
# Pour les mails non spams, des terms se rapprochent tels que need, will, information mais aussi plus informatif tel que work, use, mail, question, how, paper


# In[22]:


# J'exporte les fréquences de mots pour une future analyse de Power BI
phishing_word_freq = pd.Series(phishing_words.split()).value_counts().head(50)
non_phishing_word_freq = pd.Series(non_phishing_words.split()).value_counts().head(50)
phishing_word_freq.to_csv('C:/Users/guilb/Documents/CV/skills/Phishing_Email_Dataset/python/phishing_word_freq.csv', header=['frequency'])
non_phishing_word_freq.to_csv('C:/Users/guilb/Documents/CV/skills/Phishing_Email_Dataset/python/non_phishing_word_freq.csv', header=['frequency'])


# In[24]:


stop_words = list(stopwords.words('english'))


# In[25]:


# Vectorisation des textes
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)
X = vectorizer.fit_transform(emails_cleaned['email_text'].fillna(''))  # Remplacer les NaN par une chaîne vide
y = emails_cleaned['is_phishing']


# In[26]:


# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[27]:


# Entraînement du modèle
model = LogisticRegression(max_iter=1000)  # Augmenter max_iter pour éviter les problèmes de convergence
model.fit(X_train, y_train)


# In[28]:


# Prédictions et évaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\nPrécision du modèle :", accuracy)
print("\nRapport de classification :\n", class_report)


# In[29]:


# Visualisation de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Phishing', 'Phishing'], yticklabels=['Non-Phishing', 'Phishing'])
plt.title("Matrice de confusion - Détection des emails de phishing")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.savefig('C:/Users/guilb/Documents/CV/skills/Phishing_Email_Dataset/python/confusion_matrix.png')
plt.show()


# In[30]:


# Interprétation des coefficients du modèle
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]
coef_df = pd.DataFrame({'Mot': feature_names, 'Coefficient': coefficients})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

print("\nMots les plus associés aux emails de phishing (top 10) :")
print(coef_df.head(10))
print("\nMots les moins associés aux emails de phishing (top 10) :")
print(coef_df.tail(10))


# In[31]:


# Visualisation des coefficients
plt.figure(figsize=(12, 6))
sns.barplot(x='Coefficient', y='Mot', data=coef_df.head(10), palette='Reds')
plt.title("Mots les plus prédictifs des emails de phishing")
plt.savefig('C:/Users/guilb/Documents/CV/skills/Phishing_Email_Dataset/python/top_phishing_words.png')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Coefficient', y='Mot', data=coef_df.tail(10), palette='Greens')
plt.title("Mots les moins prédictifs des emails de phishing")
plt.savefig('C:/Users/guilb/Documents/CV/skills/Phishing_Email_Dataset/python/top_non_phishing_words.png')
plt.show()


# In[32]:


# Exporter les résultats pour Power BI
coef_df.to_csv('C:/Users/guilb/Documents/CV/skills/Phishing_Email_Dataset/python/coefficient_analysis.csv', index=False)


# In[33]:


pip install ipynb-py-convert


# In[2]:


jupyter nbconvert --to script phishing_emails.ipynb


# In[ ]:




