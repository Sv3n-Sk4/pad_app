import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

import math

import pickle
import lightgbm as lgb
import shap

st.write("# App Prêt à Dépenser")

st.image(
            "https://raw.githubusercontent.com/Sv3n-Sk4/pad_app/main/Images/PAD.png", 
            use_column_width=True
        )

st.write("""Cette app est une aide à la décision en matière d'octroi de crédit.
Elle cherche à prédire si un emprunteur pourrait présenter une défaillance.

Les données sont disponibles via ce [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) et fournies par Home Credit.
""")

st.sidebar.header('Données client à utiliser')

st.sidebar.markdown("""
[Exemple de fichier CSV à renseigner](https://raw.githubusercontent.com/Sv3n-Sk4/pad_app/main/data_exemple.csv) 
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload de votre fichier CSV", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    url="https://raw.githubusercontent.com/Sv3n-Sk4/pad_app/main/data_exemple.csv"
    input_df = pd.read_csv(url)

# Combines l'input utlisateur avec le dataset complet pour la phase d'encodage
# model_data = "https://media.githubusercontent.com/media/Sv3n-Sk4/pad_app/main/model_data.csv"
model_data = "https://raw.githubusercontent.com/Sv3n-Sk4/pad_app/main/model_data.csv"
data_raw = pd.read_csv(model_data)
data = data_raw.drop(columns=['TARGET'])

common_columns = [col for col in input_df.columns if col in data.columns]
input_df = input_df[common_columns]
df = pd.concat([input_df,data],axis=0)

# Import des librairies pour l'encodage
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def feat_engi(df):
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'] 
    return df
  
df_efe = feat_engi(df)
data_display = feat_engi(data_raw)

df = df_efe[:1] # Selection du premier individu (l'input utlisateur)

# Affichage des variables de l'input
st.subheader('Données client')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write("Attente de l'upload d'un fichier csv.") 
    st.write("Dans l'attente d'un upload, affichage de l'exemple:")
    st.write(input_df)


# Affichage des différentes informations client-emprunteur
st.write("## Renseignements clients-emprunteurs")

# Partie sur le sexe du client-emprunteur
st.write("### Sexe du client-emprunteur")
    
if input_df['CODE_GENDER'][0] == "F":
    st.write("Le client est de sexe féminin.")
elif input_df['CODE_GENDER'][0] == "M":
    st.write("Le client est de sexe masculin.")
else:
   st.write("Le sexe du client n'est pas renseigné.") 

# data_graph = pd.read_csv('https://raw.githubusercontent.com/Sv3n-Sk4/pad_app/main/model_data.csv')
data_graph = pd.read_csv(model_data)

# Création du graphique de défaillance selon le sexe du client
data_graph = data_graph[-data_graph["CODE_GENDER"].isin(["XNA"])]
pal = ["royalblue", "lightcoral"]
dfcross = pd.crosstab(data_graph['CODE_GENDER'], data_graph['TARGET']).apply(lambda r: r/r.sum()*100, axis=1)
ax = dfcross.plot.bar(figsize=(10,7),stacked=True, rot=0, color=pal)
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title="Défaillance", labels=['Non Défaillant', 'Défaillant'])
ax.set_xlabel('Sexe')
ax.set_ylabel('Pourcentage')
for rec in ax.patches:
    height = rec.get_height()
    ax.text(rec.get_x() + rec.get_width() / 2, 
                  rec.get_y() + height / 2,
                  "{:.0f}%".format(height),
                  ha='center', 
                  va='bottom')
fig = ax.get_figure()
st.pyplot(fig)


# Partie sur l'âge du client-emprunteur
st.write("### Age du client-emprunteur")

age = math.floor((input_df['DAYS_BIRTH'] / -365))
st.write("Notre client(e) est âgé(e) de : ", age, " ans.") 

# Création du premier graph de l'âge
fig, ax = plt.subplots(figsize=(12,10))
sns.kdeplot(data_graph.loc[data_graph['TARGET'] == 0, 'DAYS_BIRTH'] / -365, label = 'target == 0', ax=ax)
sns.kdeplot(data_graph.loc[data_graph['TARGET'] == 1, 'DAYS_BIRTH'] / -365, label = 'target == 1', ax=ax) 
plt.axvline(age, color='indianred', ls='--')
ax.set_xlabel('Age (années)') 
ax.set_ylabel('Densité')
ax.set_title('Distribution des âges')
legend=['Absence de défaillance','Défaillance']
ax.legend(legend, facecolor='w', loc='upper right', bbox_to_anchor=(1.3, 1))
fig = ax.get_figure()
st.pyplot(fig)

#Preprocess du second graph de l'âge
# Scinde dans un autre dataframe l'information de l'âge 
df_age = data_graph[['TARGET', 'DAYS_BIRTH']]
df_age['YEARS_BIRTH'] = df_age ['DAYS_BIRTH'] / -365
# Découpage des catégories
df_age['YEARS_BINNED'] = pd.cut(df_age['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
# Calcul de la moyenne
df_age_groups  = df_age.groupby('YEARS_BINNED').mean()

age_check = pd.cut([age], bins = np.linspace(20, 70, num = 11))
df_age_groups['AGE_CHECK'] = df_age_groups.index == age_check[0]

# Création du second graph de l'âge
fig, ax = plt.subplots(figsize=(12,10))
ax.bar(df_age_groups.index.astype(str), 100 * df_age_groups['TARGET'], color=df_age_groups.AGE_CHECK.map({True: 'indianred', False: 'lightblue'}))
ax.set_xlabel("Groupe d'âges (années)")
ax.set_ylabel('Défaillance à rembourser (%)')
ax.set_title("Défaillance à rembourser par groupe d'âges")
fig = ax.get_figure()
st.pyplot(fig)


# Partie sur le taux d'emploi
st.write("### Taux d'emploi")

employed_percentage = input_df['DAYS_EMPLOYED'] / input_df['DAYS_BIRTH']
st.write("Notre client(e) à travailler", round(employed_percentage[0]*100, 2), " % de sa vie.") 

anom = data_graph[data_graph['DAYS_EMPLOYED'] == 365243]
non_anom = data_graph[data_graph['DAYS_EMPLOYED'] != 365243]
# Creation d'un avertissement d'anomalie
data_graph['DAYS_EMPLOYED_ANOM'] = data_graph["DAYS_EMPLOYED"] == 365243
# Remplacement des outliers par un nan
data_graph['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)  
data_graph['DAYS_EMPLOYED_PERCENT'] = (data_graph['DAYS_EMPLOYED'] / data_graph['DAYS_BIRTH'])*100
data_graph['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)    
# Scinde dans un autre dataframe l'information de l'âge 
employed_data = data_graph[['TARGET', 'DAYS_EMPLOYED_PERCENT']]
employed_data['PERCENT_EMPLOYED'] = employed_data['DAYS_EMPLOYED_PERCENT']*-100
# Découpage des catégories
employed_data['EMPLOYED_BINNED'] = pd.cut(employed_data['DAYS_EMPLOYED_PERCENT'], bins = np.linspace(0, 70, num = 8))
employed_groups  = employed_data.groupby('EMPLOYED_BINNED').mean()
emp_per = round(employed_percentage[0]*100, 2)
employed_check = pd.cut([emp_per], bins = np.linspace(0, 70, num = 8))
employed_groups['EMPLOY_CHECK'] = employed_groups.index == employed_check[0]

# Création du graph de l'employabilité
fig, ax = plt.subplots(figsize=(12,10))
ax.bar(employed_groups.index.astype(str), 100 * employed_groups['TARGET'], color=employed_groups.EMPLOY_CHECK.map({True: 'indianred', False: 'lightblue'}))
ax.set_xlabel("Pourcentage de vie dans l'emploi (%)")
ax.set_ylabel('Défaillance à rembourser (%)')
ax.set_title("Défaillance à rembourser par groupe de taux de travail effectué")
fig = ax.get_figure()
st.pyplot(fig)


# nettoyage des graphs
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()
fig.clear(True)
ax.clear()  

# Lecture du pipeline
load_clf = pickle.load(open('data_clf.pkl', 'rb'))

# Application pour prédictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

# Affichage des prédictions du modèle
st.subheader('Prédiction')
defaillance = np.array(['0','1'])
st.write(defaillance[prediction])

st.subheader('Probabilités')
st.write(prediction_proba)


st.write("### Interprétabilité de la prédiction")
st.write("""Afin de mieux comprendre les résultats obtenus et d'expliquer au client-emprunteur la prédiction, vous pouvez cliquer sur le bouton ci-dessous pour afficher des graphiques d'aide à l'interprétation.
(Attention l'exécution de cette action peut prendre jusqu'à une trentaine de minutes.)
""")

graph_display = st.button("Affichage des graphiques d'interprétation")

if graph_display:
#     # Lecture du modèle seul (LIME et SHAP fonctionnant mal avec un pipeline)
#     model_lgbm = pickle.load(open('model_lgbm.pkl', 'rb'))

#     # Utilisation de la partie graphique à l'aide de Pycaret (usuellement utilisé pour sélectionner des modèles)
#     from pycaret.classification import * 
#     df_lgbm = feat_engi(data_raw)
#     clf = setup(df_lgbm, target='TARGET', session_id=42, silent=True, fix_imbalance=True)
#     lgbm2 = create_model(model_lgbm, cross_validation = False)

#     # Retirer les commentaires pour nouveau plot
#     plot_model(lgbm2, plot = 'feature', save=True)

#     st.write("")
#     st.write("Voici les variables les plus importantes de notre modèle : ") 
#     st.write("")
#     st.image(
#                 "Feature Importance.png", 
#                 use_column_width=True
#             )

    st.write("")
    st.write("Voici les variables les plus importantes de notre modèle : ") 
    st.write("")
    st.image(
                "https://raw.githubusercontent.com/Sv3n-Sk4/pad_app/main/Images/Feature Importance.png",
                use_column_width=True
            )


#     # INTEGRATION DE LIME ET SHAP
            
#     datawithoutnan = "https://raw.githubusercontent.com/Sv3n-Sk4/pad_app/main/datawithoutnan25.csv"
#     data_lime = pd.read_csv(datawithoutnan)

#     common_columns = [col for col in input_df.columns if col in data_lime.columns]
#     input_df = input_df[common_columns]
#     dflime = pd.concat([input_df,data_lime],axis=0)

#     dflime_efe = feat_engi(dflime)
#     # dfl = dflime_efe[:1] # Selects only the first row (the user input data)

#     def cat_list(df):
#         dummies = []

#         for col in df.columns:
#             if df[col].dtype == 'object':
#                     dummies.append(str(col))
            
#         return dummies

#     dummies = cat_list(dflime_efe)

#     encoded = pd.get_dummies(dflime_efe[dummies], drop_first=True)
#     dflime_efe = dflime_efe.drop(dummies, axis=1)
#     dflime_efe = dflime_efe.join(encoded)

#     dfl = dflime_efe[:1] # Selects only the first row (the user input data)
#     dfl = dfl.drop(['TARGET'], axis=1)

#     dflime_efe = dflime_efe.fillna(dflime_efe.mean())
    
#     # #Supprimer les premiers rangs redondants
#     dflime_efe = dflime_efe.iloc[3:]

#     from sklearn.model_selection import train_test_split

#     import re
#     dflime_efe = dflime_efe.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

#     y = dflime_efe['TARGET']
#     X = dflime_efe.drop(['TARGET'], axis=1)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     model_lgbm.fit(X_train, y_train)

#     test_1 = dfl.iloc[0]
    
#     shap_explainer = shap.TreeExplainer(model_lgbm)
#     shap_values = shap_explainer.shap_values(X)
    
#     # nettoyage des graphs
#     plt.figure().clear()
#     plt.close()
#     plt.cla()
#     plt.clf()
#     fig.clear(True)
#     ax.clear() 

#     shap.summary_plot(shap_values, X_train, plot_size=(20,20), show=False)
#     plt.savefig("summary_plot.png")
#     plt.close()

#     st.write("")
#     st.write("Le Summary Plot : ") 
#     st.write("")
#     st.image(
#                 "summary_plot.png",
#                 use_column_width=True
#             )

    st.write("")
    st.write("Voici les variables les plus importantes de notre modèle : ") 
    st.write("")
    st.image(
                "https://raw.githubusercontent.com/Sv3n-Sk4/pad_app/main/Images/summary_plot.png",
                use_column_width=True
            )

#     import lime 
#     from lime import lime_tabular

#     lime_explainer = lime_tabular.LimeTabularExplainer(
#         training_data=np.array(X_train),
#         feature_names=X_train.columns,
#         class_names=['Non Défaillant', 'Défaillant'],
#         mode='classification'
#     )

#     lime_exp = lime_explainer.explain_instance(
#         data_row=test_1,
#         predict_fn=model_lgbm.predict_proba
#     )

#     lime_exp.save_to_file('limeexport.html')

#     import streamlit.components.v1 as components

#     st.write("")
#     st.write("Analyse Lime du client : ") 
#     st.write("")

#     HtmlFile = open("limeexport.html", 'r', encoding='utf-8')
#     source_code = HtmlFile.read() 
#     print(source_code)
#     components.html(source_code)

#     st.write("")
#     st.write("Analyse Lime du client : ") 
#     st.write("")   

    st.write("")
    st.write("Voici les variables les plus importantes de notre modèle : ") 
    st.write("")
    st.image(
                "https://raw.githubusercontent.com/Sv3n-Sk4/pad_app/main/Images/lime.png",
                use_column_width=True
            )
            
#     HtmlFile = open("https://raw.githubusercontent.com/Sv3n-Sk4/pad_app/main/Images/limeexport.html", 'r', encoding='utf-8')
#     source_code = HtmlFile.read() 
#     print(source_code)
#     components.html(source_code)

#     def st_shap(plot, height=None):
#         shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
#         components.html(shap_html, height=height)

#     st.write("")
#     st.write("Analyse SHAP du client : ") 
#     st.write("")

#     st_shap(shap.force_plot(shap_explainer.expected_value[0], shap_values[0][1, :], test_1))

    st.write("")
    st.write("Voici les variables les plus importantes de notre modèle : ") 
    st.write("")
    st.image(
                "https://raw.githubusercontent.com/Sv3n-Sk4/pad_app/main/Images/shap.PNG",
                use_column_width=True
            )
