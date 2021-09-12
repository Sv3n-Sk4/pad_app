# PAD_APP

Le dossier contient 2 fichiers .py. Le premier pad_app.py correspond au code de l'application déployé grâce à Streamlit, le econd model_building.py permet d'avoir un aperçu de la création du modèle de classification utilisé.

Sont également présents des fichiers .csv, correspondant aux fichiers de Home Credit mais réduits compte tenu des contraintes fixées par les plateformes gratuites utilisées.
Pour autant la modélisation a été effectuée sur les jeux de données complet, et a été exportée au format .pkl

Enfin sont présent les images pour l'application, les requirements et setup pour permettre la bonne exécution de cette version de démonstration en ligne (limitée compte tenu des contraintes abordées précédemment.

L'application est disponible à l'adresse suivante :

https://share.streamlit.io/sv3n-sk4/pad_app/main/pad_app.py

L'application a été développée dans une optique entreprise (PAD) - conseiller - client.
Le but est ici d'intégrer par le conseiller, les données d'un client via l'apport d'un fichier csv (ou en le selectionnant dans un menu déroulant) puis l'application s'exécute pour donner la prédiction de solvabilité et des interprétations pour permettre au conseiller d'offrir des explications au client quant au résultat obtenu.
