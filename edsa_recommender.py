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

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview"]

    """Movie Recommender App with Streamlit """
    st.sidebar.image('resources/imgs/ReelInsights logo.jpeg', use_column_width=True)
    st.sidebar.markdown('Welcome to ReelInsights, your go-to destination for discovering fresh cinematic gems inspired by your all-time favorites!')
    st.sidebar.markdown('    ')

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Home page", "Project summary","Explore the Data", "Recommender System","Solution Overview","Meet the Team", "Contact Us"]

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
        st.write("Our winning approach")
        st.image('resources/imgs/contentVScollab2.png', width= 250, use_column_width=True)
        st.markdown("We developed our recommendation system with a strong emphasis on user-friendliness, aiming to provide an optimal user experience. Our goal is to expose users to both familiar and new content. This unique approach ensures that our Recommender System not only attracts more customers due to its user-friendly design but also promotes customer retention. This in turn leads to increased subscription rates as users seek the best movie recommendations, ultimately leading to higher profitability for the company.")
        st.markdown("Moreover, our system will put Netflix at the forefront of the competition among platforms with similar content as it is specifically designed for personalized movie recommendations based on user preferences and behavior, our movie recommender system incorporates collaborative filtering and content-based filtering algorithms, considering user ratings and similarities between users.")
        st.markdown("To further improve recommendation quality, we implemented feature engineering techniques including merging data sets, determining the average ratings and top users, amongst others these strategies guarantee a great model performance.")
        st.markdown("We further evaluated our recommender system using RMSE score which was 0.82 for our top perfoming model which is SVD model. This metric helps us measure the accuracy and effectiveness of our recommendations.")


    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    
    # Building our the "Home" page
    if page_selection == "Home page":
        #st.title("Movie Recommender System")
        st.title("Welcome To ReelInsights")
        st.markdown("***ReelInsights: Elevating Your Movie Experience Beyond Imagination!***")
        st.image('resources/imgs/possible homapage.jpg',width= 350,use_column_width=True)

    # Building our "Project Summary" page
    if page_selection == "Project summary":
       st.title("Project summary")
       st.image('resources/imgs/possible homapage.jpg', width= 250, use_column_width=True) 
       st.markdown('Our client *Netflix* is a subscription-based streaming service available in over 190 countries with over 200 billion subscribers worldwide. It provides a wide variety of TV shows, movies, documentaries, and original content across various genres. Netflix is known for its user-friendly interface and for producing original quality content. Due to the streaming landscape becoming increasingly competitive with the emergence of new platforms, Netflix is looking for a new movie recommender system to assit in the need to continually attract and retain subscribers.')
       st.markdown('Wise Analytics was tasked to delevop a movie recommender system called ReelInsights that will enable Netflix to retain and attract new subscribers by analyzing user preferences, viewing history, and movie ratings to offer personalized movie suggestions. This enhances user satisfaction by helping them discover content that aligns with their tastes and  increases engagement, longer viewing sessions, and higher retention rates for Netflix')
        

    # Bulding our the "Explore the Data" page
    if page_selection == "Explore the Data":
        st.title("Movie Recommender System")
        st.subheader ("**Exploratory Data Analysis**")
        st.markdown("**The distribution of ratings in the dataset**")
        st.image('resources/imgs/Distribution of ratings.PNG',width= 250,use_column_width=True)
        st.markdown("This visual displays ratings given by users to movies lies in between 0.5 to 5 with a high proportion of the movies have been rated 3 or 4 by the users. The distribution of ratings look a bit left skewed as large proportion of ratings is in between 3 to 5.")
        st.markdown("**Distribution of Genres in the dataset**")
        st.image('resources/imgs/distribution of genres.JPG',width= 250,use_column_width=True)
        st.markdown("There are 19 different genres of movies including Drama, Comedy, Action and Thriller being the top 4 genres of movies present in the dataset and many others.")
        st.markdown("**Yearly ratings**")
        st.image('resources/imgs/yearly total ratings.JPG',width= 250,use_column_width=True)
        st.markdown('There is a decrease in the number of ratings post 2004, this could be due to the fact that subscribers are not getting recommended new movies to watch')
        st.markdown("")



    # Bulding our the "Meet the Team" page
    if page_selection == "Meet the Team":
        st.title("Meet the Team")
        st.image('resources/imgs/Meet the team.PNG',width= 250,use_column_width=True)
        #st.markdown(" * **Fabian Dafat** : Team Lead  ")
        #st.image('resources/imgs/Tshiamo.jpeg',width= 250,use_column_width=False)
        #st.markdown(" * **Tshiamo Malebo** : Project manager ")
        #st.image('resources/imgs/Desiree.jpeg',width= 250,use_column_width=False)
        #st.markdown(" * **Desiree Malebana** : ML Engineer ")
        #st.image('resources/imgs/Boitumelo.jpeg',width= 250,use_column_width=False)
        #st.markdown(" * **Boitumelo Lefophana** :  Full Stack Data Analyst")
        #st.markdown(" * **Victoria Mohale** : Data scientist  ")
        #st.markdown(" * **Maria Boysen** : Business Analysit  ")

    # Bulding our the "Contact Us" page
    if page_selection == "Contact Us":
        st.title('Contact Us')
        st.image('resources/imgs/Contact us - Copy - Copy.webp',width= 250,use_column_width=False)
        st.markdown('* Tel: 012 6730 391')
        st.markdown('* LinkedIn: ReelInsight')
        st.markdown('* Twitter: @ReelInsights')
        st.markdown('* Instagram: @ReelInsights')
        st.markdown('* Address: 11 Adriana Cres, Rooihuiskraal, Centurion, 0154')


        
if __name__ == '__main__':
    main()
