# Importation des librairies
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write('''
# App Simple pour la prévision des fleurs d'Iris
Cette application prédit la catégorie des fleurs d'Iris
''')

st.sidebar.header("Les parametre d'entrée")
# Creation d'une fonction
def user_input():
    sepal_length = st.sidebar.slider('la longeur du Petal', 4.3, 7.9, 5.3)
    sepal_width = st.sidebar.slider('la largeur du Petal', 2.0, 4.4, 3.3)
    petal_length = st.sidebar.slider('la longeur du Petal', 1.0, 6.9, 2.3)
    petal_width = st.sidebar.slider('la largeur du Petal', 0.1, 2.5, 1.3)

# Mettre les paramètres a etudier dans un dictionnaire data
    data={'sepal_length':sepal_length,
    'sepal_width':sepal_width,
    'petal_length':petal_length,
    'petal_width':petal_width
    }
# Mettre les parametres dans un DataFrame
    fleur_parametres=pd.DataFrame(data, index=[0])
    return fleur_parametres

df=user_input()

st.subheader('on veut trouver la catégorie de cette fleur')
st.write(df)

# Utilisation du ML
iris=datasets.load_iris()
clf = RandomForestClassifier()
clf.fit(iris.data, iris.target)

# Faire des prédictions
prediction=clf.predict(df)

st.subheader("La catégorie de la fleur d'Iris est :")
st.write(iris.target_names[prediction])

