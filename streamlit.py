#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:35:58 2024

@author: raphael.aulas
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import folium

import colormaps
import branca.colormap as cm
from colormap import rgb2hex
from matplotlib import colors
import matplotlib.colors as mcolors
from streamlit_folium import st_folium
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
# import scikit 
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
from sklearn.cluster import AgglomerativeClustering
# import plotly.figure_factory as ff



#préparation datas 0
# @st.cache_data
path = './'
name = '2024_52.csv'

datas = pd.read_csv(path + name , index_col= 0)

path = './'
name = '52.geojson'
carto = gpd.read_file(path + name)
carto.INSEE_COM = carto.INSEE_COM.astype('int')
# display ( carto )

gdf = pd.merge(carto, datas, left_on='INSEE_COM', right_on='INSEE_COM', how='left')
gdf = gdf.drop( ['ID', 'NOM_M'], axis = 1 )
col_selectionnables = [  'POP_Total',
 'L24T1_COM_ratio', 'L24T1_DIV_ratio', 'L24T1_DSV_ratio',  'L24T1_DVC_ratio',  'L24T1_DVD_ratio',  'L24T1_DVG_ratio',  'L24T1_ECO_ratio',
 'L24T1_ENS_ratio',  'L24T1_EXD_ratio', 'L24T1_EXG_ratio', 'L24T1_FI_ratio', 'L24T1_HOR_ratio', 'L24T1_LR_ratio', 'L24T1_RDG_ratio',
 'L24T1_REC_ratio', 'L24T1_REG_ratio', 'L24T1_RN_ratio', 'L24T1_SOC_ratio', 'L24T1_UDI_ratio', 'L24T1_UG_ratio', 'L24T1_UXD_ratio',
 'L24T1_VEC_ratio', 'L24T1_Abstentions_ratio',
 'DIP_BAC_+_2_ratio',
 'DIP_BAC_+_3_ou_4_ratio',
 'DIP_BAC_+_5_ratio',
 'DIP_BAC,_brevet_pro_ou_équiv_ratio',
 'DIP_Brevet_des_collèges_ratio',
 'DIP_CAP-BEP_ou_équiv_ratio',
 'DIP_Sans_diplôme_ou_CEP_ratio',
'CHO_Chôm_15-24_ans_ratio',
 'CHO_Chôm_25-54_ans_ratio',
 'CHO_Chôm_55-64_ans_ratio']
# Sélection des colonnes avec des valeurs supérieures à 0
col_affichables = [col for col in col_selectionnables if (gdf[col] > 0).any()]
# print ( gdf.columns[col_affichables] )

# @st.cache_data

#MACHINE

st.title("Analyse d'un territoire")
st.sidebar.title("Sommaire")
pages=["Introduction", "Visualisation", "Cartographie des résultats électoraux","Corrélations", "classification opérationnelle",  "conclusions"]
page=st.sidebar.radio("Aller vers", pages)
if page == pages[0] : 
    st.write("### Introduction")
    with st.container():
        st.write("Ce document est une esquisse d'analyse territoriale basée sur la géomatique et l'analyse statistique. Le territoire analysé est le département de la Haute-Saône, coeur de la ruralité heureuse et épicentre du séisme politique de l'année 2024 qui a vu les partis d'extrème-droite atteindre des scores inégalés depuis un siècle.")
        st.write ( "La série de données intégre : les résultats électoraux des européennes et du premier tour des législatives  de 2024, les données INSEE représentant le niveau de diplôme et de chomage par tranches d'age de la population des villes qui composent ce département.")
        
        st.write( "Vous pouvez trouver une analyse des données, une représentation des données, une représentation des résultats electoraux, et enfin une anlyse des corrélations se terminant par une clusterisation des villes.")

        
        # # Bonus : Visualisation de la distribution (pour les colonnes numériques)
        # if datas[selected_column].dtype in ['int64', 'float64']:
        #     st.subheader(f"Distribution de la colonne '{selected_column}'")
        #     st.histogram(gdf[selected_column])
            
    with st.container():
        st.write("Ceci est le conteneur intérieur.")
if page == pages[1] : 
    st.write("### Visualisation")
    # Sélection de la colonne
    selected_column = st.selectbox("Choisissez une colonne à analyser", col_affichables )
    
    # Affichage des 10 premières lignes triées
    st.subheader(f"10 premières lignes de la colonne '{selected_column}' (triées)")
    sorted_df = gdf.sort_values(by=selected_column, ascending=False)
    st.dataframe(sorted_df[[selected_column]].head(10))
    
    # Affichage des statistiques descriptives
    st.subheader(f"Statistiques descriptives de la colonne '{selected_column}'")
    st.dataframe(gdf[selected_column].describe())
    
    
    # Visualisation de la distribution (pour les colonnes numériques)
    st.subheader(f"Distribution de la colonne '{selected_column}'")
    fig, ax = plt.subplots()
    ax.hist(gdf[selected_column], bins=20, edgecolor='black')
    ax.set_title(f"Histogramme de {selected_column}")
    ax.set_xlabel(selected_column)
    ax.set_ylabel("Fréquence")
    st.pyplot(fig)

if page == pages[2] : 
    st.write("### cartographie électorale")
    # Sélection de la colonne

    #choix liste
    d_ch_liste =(
    'E24_Voix_1',    'E24_Voix_2' ,
    'E24_ZEMMOUR' ,
    'E24_LFI' ,
    'E24_RN' ,
    'E24_EELV' ,
    'E24_Voix_7' ,
    'E24_Voix_8' ,
    'E24_Voix_9' ,
    'E24_Voix_10' ,
    'E24_LREM',
    'E24_Voix_12',
    'E24_Voix_13',
    'E24_Voix_14',
    'E24_Voix_15',
    'E24_Voix_16',
    'E24_Voix_17',
    'E24_LR',
    'E24_Voix_19',
    'E24_Voix_20',
    'E24_Voix_21',
    'E24_Voix_22',
    'E24_Voix_23',
    'E24_Voix_24',
    'E24_Voix_25',
    'E24_Voix_26',
    'E24_PSPP',
    'E24_Voix_28',
    'E24_Voix_29',
    'E24_Voix_30',
    'E24_Voix_31',
    'E24_Voix_32',
    'E24_PCF',
    'E24_Voix_34',
    'E24_Voix_35',
    'E24_Voix_36',
    'E24_Voix_37',
    'E24_Voix_38', 
    'L24T1_COM', 
    'L24T1_DIV', 
    'L24T1_DSV' ,
    'L24T1_DVC', 
    'L24T1_DVD', 
    'L24T1_DVG',
    'L24T1_ECO', 
    'L24T1_ENS',
    'L24T1_EXD',
    'L24T1_EXG',
    'L24T1_FI',
    'L24T1_HOR',
    'L24T1_LR',
    'L24T1_RDG',
    'L24T1_REC', 
    'L24T1_REG', 
    'L24T1_RN', 
    'L24T1_SOC', 
    'L24T1_UDI',
    'L24T1_UG', 
    'L24T1_UXD', 
    'L24T1_VEC'
)


# Fonction pour créer le dictionnaire
    def create_dict_from_list(data_list, election ):
        result_list = []
        for item in data_list:
            print ( item)
            if '_' in item:
                value,key = item.split('_', 1)  # Split only on the first occurrence of '_'
                if value == election:
                    result_list.append( key)
            # else:
            # st.warning(f"L'élément '{item}' ne contient pas de séparateur '_' et sera ignoré.")
        return result_list
    
    # choix Elections
    ch_election = [ "2024Europenne","2024LEG_tour1" ]
    d_ch_election = { "2024Europenne"  : "E24", "2024LEG_tour1":"L24T1" }
    selected_column = st.selectbox("Choisissez une élection à analyser", ch_election )
    suff_viz = d_ch_election[selected_column]
    # st.write(suff_viz)
    viz = st.selectbox("Choisissez une liste à cartographier",create_dict_from_list(d_ch_liste,suff_viz )  )
    # st.write(viz)
    viz =  suff_viz +'_'+ viz
    
    viz_ratio = viz + "_ratio"
    alias_viz = "pourcentage"
    color_liste = {
    'E24_Voix_1': 'r',
    'E24_Voix_2': 'r',
    'E24_ZEMMOUR': 'maroon',
    'E24_LFI': 'darkred',
    'E24_RN': 'olive',
    'E24_EELV': 'darkgreen',
    'E24_Voix_7': 'r',
    'E24_Voix_8': 'r',
    'E24_Voix_9': 'r',
    'E24_Voix_10': 'r',
    'E24_LREM': 'mediumturquoise',
    'E24_Voix_12': 'r',
    'E24_Voix_13': 'r',
    'E24_Voix_14': 'r',
    'E24_Voix_15': 'r',
    'E24_Voix_16': 'r',
    'E24_Voix_17': 'r',
    'E24_LR': 'blue',
    'E24_Voix_19': 'r',
    'E24_Voix_20': 'r',
    'E24_Voix_21': 'r',
    'E24_Voix_22': 'r',
    'E24_Voix_23': 'r',
    'E24_Voix_24': 'r',
    'E24_Voix_25': 'r',
    'E24_Voix_26': 'r',
    'E24_PSPP': 'deeppink',
    'E24_Voix_28': 'r',
    'E24_Voix_29': 'r',
    'E24_Voix_30': 'r',
    'E24_Voix_31': 'r',
    'E24_Voix_32': 'r',
    'E24_PCF': 'red',
    'E24_Voix_34': 'r',
    'E24_Voix_35': 'r',
    'E24_Voix_36': 'r',
    'E24_Voix_37': 'r',
    'E24_Voix_38': 'r', 
    'L24T1_COM': 'r', 
    'L24T1_DIV': 'r', 
    'L24T1_DSV': 'r' ,
    'L24T1_DVC': 'blue', 
    'L24T1_DVD': 'r', 
    'L24T1_DVG': 'orchid',
    'L24T1_ECO': 'r', 
    'L24T1_ENS': 'steelblue',
    'L24T1_EXD': 'r',
    'L24T1_EXG': 'lightcoral',
    'L24T1_FI': 'brown',
    'L24T1_HOR': 'r',
    'L24T1_LR': 'blue',
    'L24T1_RDG': 'r',
    'L24T1_REC': 'r', 
    'L24T1_REG': 'r', 
    'L24T1_RN': 'olive', 
    'L24T1_SOC': 'deeppink', 
    'L24T1_UDI': 'deepskyblue',
    'L24T1_UG': 'orangered', 
    'L24T1_UXD': 'maroon', 
    'L24T1_VEC': 'darkgreen'
    }
  
    p_min = gdf[viz_ratio].min() 
    p_max = gdf[viz_ratio].max()

    # Définir les couleurs de départ et d'arrivée
    start_color = 'white'
    # end_color = 'maroon'
    end_color = color_liste[viz]
    # Créer une fonction de mappage de couleurs
    color_ramp = mcolors.LinearSegmentedColormap.from_list("custom_ramp", [start_color, end_color])
    num_colors = 7

    color_list = [color_ramp(i) for i in np.linspace(0, 1, num_colors)]
    # Convertir les couleurs en codes hexadécimaux
    hex_colors = [mcolors.to_hex(color) for color in color_list]

    # Afficher les codes hexadécimaux
    # for i, color in enumerate(hex_colors):
    #     print(f"Couleur {i+1}: {color}")





    colormap = cm.StepColormap( 
                                colors= hex_colors , 
                                vmin=p_min, 
                                vmax=p_max,  
                                #   index=bornes,
                                # caption='step'
                                )

    m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], 
               zoom_start=10)


    folium.GeoJson(
                    gdf,
                    style_function=lambda feature: {
                        'fillColor': colormap(feature['properties'][viz_ratio]),
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7
                    },
                    tooltip=folium.GeoJsonTooltip(fields=[ 'NOM', viz_ratio],
                                                  aliases=['Ville :', 'pct %'],
                                                  localize=True)
                    ).add_to(m)

    # Ajout de la légende
    # colormap.add_to(m)
    colormap.caption = 'pourcentage du vote '+viz
    colormap.add_to(m)
     # Affichage des statistiques descriptives
    st.subheader(f"Statistiques descriptives de la liste '{viz}'")
    st.dataframe(gdf[viz_ratio].describe())
    
    #   Folium dasn  Streamlit
    st_data = st_folium(m, width=1000)


if page == pages[3] : 
    st.write("### corrélations")
    gdf_corr = gdf[['L24T1_Abstentions_ratio' ,'L24T1_ENS_ratio' ,'L24T1_RN_ratio' ,'L24T1_UG_ratio' , 'CHO_Chôm_15-24_ans_ratio' , 'CHO_Chôm_25-54_ans_ratio' , 'CHO_Chôm_55-64_ans_ratio']]
    # # Créer une figure avec une taille personnalisée
    # fig = plt.figure(figsize=(20, 20))  # Ajustez ces valeurs selon vos besoins
    
    # # Créer la scatter_matrix
    # # scatter_matrix(gdf_corr, ax=fig.gca(), figsize=(20, 20))
    # st.scatter_chart(gdf_corr )


    import altair as alt
    # from vega_datasets import data

    # source = data.cars()

    chart = alt.Chart(gdf_corr).mark_circle().encode(
        alt.X(alt.repeat("column"), type='quantitative'),
        alt.Y(alt.repeat("row"), type='quantitative'),
        color='Origin:N'
    ).properties(
        width=150,
        height=150
    ).repeat(
        row=['L24T1_Abstentions_ratio' ,'L24T1_ENS_ratio' ,'L24T1_RN_ratio' ,'L24T1_UG_ratio' , 'CHO_Chôm_15-24_ans_ratio' , 'CHO_Chôm_25-54_ans_ratio' , 'CHO_Chôm_55-64_ans_ratio'],
        column=['L24T1_Abstentions_ratio' ,'L24T1_ENS_ratio' ,'L24T1_RN_ratio' ,'L24T1_UG_ratio' , 'CHO_Chôm_15-24_ans_ratio' , 'CHO_Chôm_25-54_ans_ratio' , 'CHO_Chôm_55-64_ans_ratio']
    ).interactive()

    # tab1, tab2 = st.tabs(["Streamlit theme (default)", "Altair native theme"])

    # with tab1:
    st.altair_chart(chart, theme="streamlit", use_container_width=True)
    # with tab2:
    #     st.altair_chart(chart, theme=None, use_container_width=True)

    st.write("### tableaux des corrélations")
    # Compute the correlation matrix
    gdf_corr = gdf[['L24T1_Abstentions_ratio' ,'L24T1_ENS_ratio' ,'L24T1_RN_ratio' ,'L24T1_UG_ratio' ,'DIP_BAC_+_2_ratio' , 'DIP_BAC_+_3_ou_4_ratio' , 'DIP_BAC_+_5_ratio' , 'DIP_BAC,_brevet_pro_ou_équiv_ratio' , 'DIP_Brevet_des_collèges_ratio' , 'DIP_CAP-BEP_ou_équiv_ratio' , 'DIP_Sans_diplôme_ou_CEP_ratio' ,'CHO_Chôm_15-24_ans_ratio' , 'CHO_Chôm_25-54_ans_ratio' , 'CHO_Chôm_55-64_ans_ratio']]


    corr = gdf_corr.corr()
    import plotly.express as px
    fig = px.imshow(corr, text_auto='.2f', aspect="auto")
    st.plotly_chart(fig, theme=None)    
    st.write( "liste des colonnes : " , gdf_corr.columns.tolist())
    st.write("Le tableau des corrélations analyse un set de données réduit à quelques variables dans un effort de facilité la lecture. Il donne néanmoins des premiers résultats intéressants : corrélation négative de la participation et des 3 listes retenues, montrant la mobilisatin de chacun des camps, décorrélation du vote RN et des diplomés à BAc+2 Bac+3, corrélation entre le vote RN et les diplomés CAP/BEP  ")
    st.write ("On prendra évidemment ces corrélations avec la plus grande prudence : 1/ il s'agit de phénomènes sociaux 2/ la granulométrie communale invite à considérer ces résultats avec humilité, la répartition des populations au sein d'une meme commune pouvant être très disparate.")
   
    # # Generate a mask for the upper triangle
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # # # Set up the matplotlib figure
    # f, ax = plt.subplots(figsize=(20, 20))
    
    # # # Generate a custom diverging colormap
    # # cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # # # Draw the heatmap with the mask and correct aspect ratio
    # # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    # #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    # heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
    # # Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
    # heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':9}, pad=12);


    # plt.show()

if page == pages[4] : 
    st.write("### classification")
    st.header("Européennes 2024 : Analyse des composantes principales")
    datas_class = datas[['INSEE_COM', 'E24_Abstentions_ratio' , 'E24_ZEMMOUR_ratio' ,'E24_LFI_ratio' ,'E24_RN_ratio' , 'E24_EELV_ratio' ,'E24_LR_ratio',  'E24_LREM_ratio' ,'E24_PSPP_ratio','E24_PCF_ratio' ]]
    # suppression des colonnes non numériques
    df_num = datas_class.drop('INSEE_COM', axis = 1  )
    
    # Normaliser les données
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_num)
    # scatter_matrix(df_num, figsize=(9, 9))
    plt.show()
    
    # PCA 
    pca = PCA()
    pca.fit(df_num)
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)
    nb_dim = len ( df_num.columns)
    eig = pd.DataFrame(
        {
            "Dimension" : ["Dim" + str(x + 1) for x in range(nb_dim)], 
            "Variance expliquée" : pca.explained_variance_,
            "% variance expliquée" : np.round(pca.explained_variance_ratio_ * 100),
            "% cum. var. expliquée" : np.round(np.cumsum(pca.explained_variance_ratio_) * 100)
        }
    )
    st.write( eig )

    # Création de la figure
    fig, ax = plt.subplots()
    
    # Création du diagramme en barres
    ax.bar(eig.Dimension, eig["% variance expliquée"])
    
    # Ajout de texte
    ax.text(5, 18, f"{round(100/nb_dim)}%")
    
    # Ajout de la ligne horizontale
    ax.axhline(y=100/nb_dim, linewidth=0.5, color="dimgray", linestyle="--")
    
    # Personnalisation du graphique (optionnel)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("% variance expliquée")
    ax.set_title("Variance expliquée par dimension")
    
    # Affichage du graphique dans Streamlit
    st.pyplot(fig)

    # 
    st.header("analyse des dimensions")
    n = df_num.shape[0] # nb individus
    p = df_num.shape[1] # nb variables
    eigval = (n-1) / n * pca.explained_variance_ # valeurs propres
    sqrt_eigval = np.sqrt(eigval) # racine carrée des valeurs propres
    corvar = np.zeros((p,p)) # matrice vide pour avoir les coordonnées
    for k in range(p):
        corvar[:,k] = pca.components_[k,:] * sqrt_eigval[k]
    # on modifie pour avoir un dataframe
    coordvar = pd.DataFrame({'id': df_num.columns, 'COR_1': corvar[:,0], 'COR_2': corvar[:,1], 'COR_3': corvar[:,2]})
    st.write( coordvar) 


# Classification ACH
    st.header("Classification par clusters")

    # Fonction pour créer le dendrogramme
    # def plot_dendrogram(X, model):
    # # Créer la matrice de liaison
       
            


    from scipy.cluster import hierarchy

    # df_scaled 
    #librairies pour la CAH
    #Afficher  les premières lignes du DataFrame
    st.subheader("Aperçu des données")
    st.write(df_num.head())

    # Paramètres pour la clusterisation
    st.write( "Notre conseil est de limiter le nombre de clusters entre 3 et 5 pour des regroupement pertinents, et de se limiter à la méthode ward plus efficace en l'espèce.")
    n_clusters = st.slider("Nombre de clusters", min_value=2, max_value=10, value=5)
    method_linkage = st.selectbox("Méthode de liaison", ["ward", "complete", "average", "single"])
    
    # Création du modèle
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method_linkage)
    
    # Fit du modèle
    model = model.fit(df_scaled)
    
    # Affichage des résultats de la clusterisation
    st.subheader("Résultats de la clusterisation")
    datas_class['Cluster'] = model.labels_
    df_num['Cluster'] = model.labels_
    st.write(df_num)
    # Statistiques par cluster
    st.subheader("Statistiques par cluster")
    for cluster in range(n_clusters):
        st.write(f"Cluster {cluster}:")
        st.write(df_num[df_num['Cluster'] == cluster].describe())
        st.write("---")
        
    # Création du dendrogramme
    plt.figure(figsize=(10, 7))
    linkage_matrix = scipy_linkage(df_scaled, method = method_linkage )
        
    # Tracer le dendrogramme
    plt.figure(figsize=(15, 15))
    

    # Trouvez la hauteur de coupure correspondant au nombre de clusters souhaité
    # threshold_dendrogram = hierarchy.fcluster(linkage_matrix  , t=n_clusters, criterion='maxclust')
    # max_d = max(linkage_matrix[:, 2])
    # threshold_dendrogram = (max_d * (n_clusters - 1)) / (len(linkage_matrix ) + 1)
    scipy_dendrogram(linkage_matrix )
    # plt.axhline(y=threshold_dendrogram, color='r', linestyle='--')
    plt.title('Dendrogramme')
    plt.xlabel('Échantillon')
    plt.ylabel('Distance')
    # plot_dendrogram(df_scaled, model)

    plt.title("Dendrogramme")
    
    # Affichez la figure dans Streamlit
    st.pyplot(plt)




    




    # # Statistiques par cluster A 
    # st.write ("Moyennes par clusters avec " + str( n_clusters)  + " clusters" )
    # st.write(datas_class.groupby('cluster').agg(['mean', 'count']))# il faut arranger ça
    
    
    gdf = pd.merge(carto, datas_class, left_on='INSEE_COM', right_on='INSEE_COM', how='left')
    # display( gdf)
    
    # Créer un mapping numérique pour les catégories
    categorie_mapping = {cat: i for i, cat in enumerate(gdf['Cluster'].unique())}
    # gdf['categorie_num'] = gdf['cluster'].map(categorie_mapping)
    
    
    # Créer une palette de couleurs
    n_categories = len(categorie_mapping)
    color_scale = cm.LinearColormap(colors=['white', 'blue', 'green', 'purple', 'orange'][:n_categories], 
                                 vmin=gdf['Cluster'].min(), 
                                 vmax=gdf['Cluster'].max())
    

    m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=9)
    
    #   Ajouter les polygones colorés à la carte
    folium.GeoJson(
        gdf,
        style_function=lambda feature: {
            'fillColor': color_scale(feature['properties']['Cluster']),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7,
        },
        tooltip=folium.GeoJsonTooltip(fields=['NOM','E24_Abstentions_ratio' , 'E24_ZEMMOUR_ratio' , 'E24_LFI_ratio' ,'E24_RN_ratio' , 'E24_EELV_ratio' ,  'E24_LREM_ratio' ,'E24_PSPP_ratio','E24_PCF_ratio', 'Cluster'], 
                                      aliases=['Ville :','abst:', 'Zemm%:', 'LFI%:', 'RN%:', 'EELV%:','LREM%:','PSPP%:','PCF%:','Cat '],
                                      style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"))
    ).add_to(m)
    
    #   Ajouter la légende
    color_scale.add_to(m)
    #   Folium dasn  Streamlit
    st_data = st_folium(m, width=1000)


if page == pages[5] : 
    st.write("### conclusions")
    st.write( "Cette présentation est une preuve de concept. Elle n'a pas pour objectif d'être opérationnelle en l'état. Pour ce faire, il faudrait étendre les données en entrée, repérer les variables les plus pertinentes lors des corrélations. Ces corrélations, pour être réalistes, doivent mieux coller à la réalité électorale, c'est à dire à l'échelon le plus petit qu'est le bureau de vote.")
    st.write("Dans sa grande sagesse, l'INSEE a construit la statistique publique sur des périmètres différents de ceux des bureaux de vote, interdisant ainsi les méthodes trop simples de recoupement.")
    st.write("Pour autant, les travaux présentés ici esquissent une méthode fiable et puissante tant pour comprendre un territoire et construire des indicateurs solides pour l'évolution des territoire. Il esquisse un travail utile :")
    st.write( "1/ Pour les exécutifs des collectivités et les responsables politiques pour aborder les prochaines échéances, ")
    st.write ( "2/ pour les administrations des collectivités qui voudraient objectiver des mécanismes complexes et trouver les clefs de compréhension pour mettre en place ou évaluer des programmes d'actions publiques,  ")
    st.write( "3/ pour les aménageurs ou acteurs de projets publics confronté aujourd'hui à des phénomènes de rejet de projets divers.")

    
    
