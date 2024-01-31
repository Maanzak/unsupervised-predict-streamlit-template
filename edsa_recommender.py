"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data handling dependencies
import pandas as pd
import numpy as np
from PIL import Image
import string 

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():
    """Movie Recommender App with Streamlit """
    st.sidebar.markdown('ReelInsights is an app that helps people find movies they like from old favourates to new discovers')
    st.sidebar.markdown('    ')

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Home page","Explore the Data","User login","Recommender System","Ratings","Solution Overview", "Meet the Team", "Contact Us"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    # Building our the "Home" page
    if page_selection == "Home Page":
        st.title("Movie Recommender System")
        st.write("Welcome")
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        st.subheader("  ")

    # Bulding our the "Explore the Data" page
    if page_selection == "Explore the Data":
        st.title("Movie Recommeder System")
        st.subheader ("**Exploratory Data Analysis**")
        st.markdown("**The distribution of ratings in the dataset")
        st.markdown("This visual displays ratings given by users to movies lies in between 0.5 to 5 with a high proportion of the movies have been rated 3, 3.5 or 4 by the users. The distribution of ratings look a bit left skewed as large proportion of ratings is in between 3 to 5.")
        st.markdown("**Distribution of Genres in the dataset**")
        st.markdown("There are 19 different genres of movies including Drama, Comedy, Action and Thriller being the top 4 genres of movies present in the dataset and many others.")
        st.markdown("**Average ratings by 15 top users**")
        st.markdown("")
        st.markdown("**Number of top 15 rated movies**")
        st.markdown("")

    # Bulding our the "User login" page
    if page_selection == "User login":
        st.title("Movie Recommender System")

    # Bulding our the "Meet the Team" page
    if page_selection == "Meet the Team":
        st.title("Movie Recommender System")
        st.markdown(" * **Fabian Dafat** : Team Lead  ")
        st.markdown(" * **Tshiamo Malebo** : Project manager ")
        st.markdown(" * **Desiree Malebana** : ML Engineer ")
        st.markdown(" * **Boitumelo Lefophana** :  Full Stack Data Analyst  ")
        st.image('resources/imgs/Boitumelo.jpeg',width= 250,use_column_width=False)
        st.markdown(" * **Boitumelo Lefophana** :  Full Stack Data Analyst")
        st.image('resources/imgs/Lesiba.jpeg',width= 250,use_column_width=False)
        st.markdown(" * **Victoria Mohale** : Data scientist  ")
        st.markdown(" * **Maria Boysen** : Business Analysit  ")

    # Bulding our the "Contact Us" page
    if page_selection == "Contact Us":
        st.markdown('Contact Us')
        st.markdown(' * Tel: 012 6730 391')
        st.markdown('* Twitter: @ReelInsights')
        st.markdown('* Instagram: @ReelInsights')
        st.markdown('* Address: ')
        st.markdown(' 11 Adriana Cres, Rooihuiskraal, Centurion, 0154')


        

if __name__ == '__main__':
    main()
