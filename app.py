# Core Packages
import streamlit as st
import altair as alt
import plotly.express as px
from PIL import Image
import base64

# EDA Packages
import pandas as pd
import numpy as np
from datetime import datetime

# Load Model
import joblib 
#pipe_lr = joblib.load(open("./models/mon_model.pkl","rb"))
pipe_lr = joblib.load(open("./models/mon_model_1.pkl","rb"))
# Track Utils
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table

# Function

#def predict_emotion(text):
#    result = pipe_lr.predict([text])

#    return result[0]

def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger" : "😠", "disgust" : "🤮", "fear" : "😨😱", "happy" : "🤗", "joy" : "😂", "neutral" : "😐", "sad" : "😔", "sadness" : "😔", "shame" : "😳", "surprise" : "😮"}
#emotions_emoji_dict = {"anger":"😠", "empty":" ", "hate":"🤮", "fun":"🤗", "enthusiasm":"🤗", "happiness":"😂", "neutral":"😐", "love":"😔", "sadness":"😔", "worry":"😳", "surprise":"😮"}

# Main Application
def main():
	st.title("My E-motion")
	menu = ["Home", "My Emotion", "Monitor", "About"]
	choice = st.sidebar.selectbox("Menu", menu)
	create_page_visited_table()
	create_emotionclf_table()
	if choice == "Home" :
		add_page_visited_details("Home", datetime.now())
		st.markdown("""
    
    	Emotions hold a paramount role in the conversation, as it expresses context to the conversation. Text/word in conversation consists of lexical 
     	and contextual meanings. Extracting emotions from text has been an interesting work recent thees years.   
    

    


    	""")

    	# image = Image.open('images/speech-text.png')
		image = Image.open('images/emo3.png')
		st.image(image)

		st.markdown("""
    	 With the advancement of machine learning techniques and hardware 
      	 to support the machine learning process, recognising emotions from a text with machine learning provides promising and significant results. 
         The underlying principle behind detecting emotional content in text is rooted in natural language processing, an expanding research domain that has gained tremendous momentum in the wake of a burgeoning volume of online comments.
    	""")

    

		st.markdown("""
    	## Find me at
    	* [Data](https://www.kaggle.com/datasets/ishantjuyal/emotions-in-text?select=Emotion_final.csv)
    	* [GitHub](https://github.com/nainiayoub)

    """)
     
	elif choice == "My Emotion":
		add_page_visited_details("My Emotions", datetime.now())
		st.markdown("""
    According to the discrete basic emotion description approach, emotions can be classified into six basic emotions: sadness, joy, surprise, anger, disgust, and fear _(van den Broek, 2013)_
    """)

		with st.form(key='emotion_clf_form'):
			text = st.text_area("Type here")
			submit = st.form_submit_button(label='Classify text emotion')
		if submit:
			if text:
				st.write(f"{text}")
				col1, col2 = st.columns(2)
           		# output prediction and proba
				prediction = predict_emotions(text)
				datePrediction = datetime.now()
				probability = get_prediction_proba(text)

				with col1:   
            		# st.write(text)
					emoji_icon = emotions_emoji_dict[prediction]
					st.success(f"Emotion Predicted : {prediction.upper()} {emoji_icon}")
            
				with col2:
					st.success(f"Confidence: {np.max(probability) * 100}%")

            		# with col2:
				st.markdown("""### Classification Probability""")
				proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
				st.write(proba_df)
            # st.write(proba_df.T)

            # plotting probability
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["emotions", "probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
				st.altair_chart(fig, use_container_width=True)
            	###### Global View ######

				if 'texts' and 'probas' and 'predictions' and 'date' not in st.session_state:
					st.session_state.texts = []
					st.session_state.predictions = []
					st.session_state.probas = []
					st.session_state.date = []

				st.markdown("""### Collecting inputs and classifications""")
            	# store text
            	# st.write("User input")
				st.session_state.texts.append(text)
            	# st.write(st.session_state.texts)

        		#store predictions
            	# st.write("Classified emotions")
				st.session_state.predictions.append(prediction.upper())
            	# st.write(st.session_state.predictions)

            	#store probabilities		
				st.session_state.probas.append(np.max(probability) * 100)

            	# store date
				st.session_state.date.append(datePrediction)

				prdcts = st.session_state.predictions
				txts = st.session_state.texts
				probas = st.session_state.probas
				dateUser = st.session_state.date


				def get_table_download_link(df):
					"""Generates a link allowing the data in a given panda dataframe to be downloaded
                	in:  dataframe
                	out: href string
                	"""
					csv = df.to_csv(index=False)
					b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
					href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
					st.markdown(href, unsafe_allow_html=True)

				if 'emotions' and 'occurence' not in st.session_state:
					st.session_state.emotions = ["ANGER", "DISGUST", "FEAR", "JOY", "NEUTRAL", "SADNESS", "SHAME", "SURPRISE"]
					#st.session_state.emotions = ["ANGER", "EMPTY", "HATE", "FUN", "ENTHUSIASM", "HAPPINESS", "NEUTRAL", "SADNESS", "LOVE", "SURPRISE", "WORRY"]
					st.session_state.occurence = [0, 0, 0, 0, 0, 0, 0, 0 ]
            

            		# Create data frame
				if prdcts and txts and probas:
					st.write("Data Frame")	
					d = {'Text': txts, 'Emotion': prdcts, 'Probability': probas, 'Date': dateUser}
					df = pd.DataFrame(d)
					st.write(df)
					get_table_download_link(df)

                ## emotions occurences
                
					index_emotion = st.session_state.emotions.index(prediction.upper())
					st.session_state.occurence[index_emotion] += 1

					d_pie = {'Emotion': st.session_state.emotions, 'Occurence': st.session_state.occurence}
					df_pie = pd.DataFrame(d_pie)
                	# st.write("Emotion Occurence")
                	# st.write(df_pie)


                	# df_occur = {'Emotion': prdcts, 'Occurence': occur['Emotion']}
                	# st.write(df_occur)

                

                	# Line chart
                	# c = alt.Chart(df).mark_line().encode(x='Date',y='Probability')
                	# st.altair_chart(c)

                

					col3, col4 = st.columns(2)
					with col3:
						st.write("Emotion Occurence")	
						st.write(df_pie)
					with col4:
						chart = alt.Chart(df).mark_line().encode(
                        x=alt.X('Date'),
                        y=alt.Y('Probability'),
                        color=alt.Color("Emotion")
                    ).properties(title="Emotions evolution by time")
					st.altair_chart(chart, use_container_width=True)

                	# Pie chart
					st.write("Probabily of total predicted emotions")
					fig = px.pie(df_pie, values='Occurence', names='Emotion')
					st.write(fig)

			else:
				st.write("No text has been submitted!")

	elif choice == "Monitor":
		add_page_visited_details("Monitor", datetime.now())
		st.subheader("Monitor App")

		with st.expander("Page Metrics"):
			page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns = ['Page Name', 'Time of Visit'])
			st.dataframe(page_visited_details)	

			pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name = 'Counts')
			c = alt.Chart(pg_count).mark_bar().encode(x = 'Page Name', y = 'Counts', color = 'Page Name')
			st.altair_chart(c,use_container_width = True)	

			p = px.pie(pg_count,values='Counts', names = 'Page Name')
			st.plotly_chart(p, use_container_width = True)

		with st.expander('Emotion Classifier Metrics'):
			df_emotions = pd.DataFrame(view_all_prediction_details(), columns = ['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
			st.dataframe(df_emotions)

			prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name = 'Counts')
			pc = alt.Chart(prediction_count).mark_bar().encode(x = 'Prediction', y = 'Counts', color = 'Prediction')
			st.altair_chart(pc, use_container_width = True)	

	else:
		add_page_visited_details("About", datetime.now())

		st.write("Welcome to the E-motion  App! This application utilizes the power of natural language processing and machine learning to analyze and identify emotions in textual data.")
		
		st.subheader("Our Mission")

		st.write("At E-motion App, our mission is to provide a user-friendly and efficient tool that helps individuals and organizations understand the emotional content hidden within text. We believe that emotions play a crucial role in communication, and by uncovering these emotions, we can gain valuable insights into the underlying sentiments and attitudes expressed in written text.")
		
		st.subheader("How It Works")
		
		st.write("When you input text into the app, our system processes it and applies deep learning model to extract meaningful features from the text. These features are then fed into the trained model, which predicts the emotions associated with the input text. The app displays the detected emotions, along with a confidence score, providing you with valuable insights into the emotional content of your text.")
		
		st.subheader("Key Features:")
		
		st.markdown("##### 1. Confidence Score")
		
		st.write("Alongside the detected emotions, our app provides a confidence score, indicating the model's certainty in its predictions. This score helps you gauge the reliability of the emotion detection results and make more informed decisions based on the analysis.")
		
		st.markdown("##### 2. User-friendly Interface")
		
		st.write("We've designed our app with simplicity and usability in mind. The intuitive user interface allows you to effortlessly input text, view the results, and interpret the emotions detected. Whether you're a seasoned data scientist or someone with limited technical expertise, our app is accessible to all.")
		
		st.subheader("Applications")
		
		st.markdown("""
	      The Emotion Detection in Text App has a wide range of applications across various industries and domains. Some common use cases include:
	      - Social media sentiment analysis
	      - Customer feedback analysis
	      - Market research and consumer insights
	      - Brand monitoring and reputation management
	      - Content analysis and recommendation systems
	      """)
		
if __name__ == '__main__':
	main()