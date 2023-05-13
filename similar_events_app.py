import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

st.title("Events Similarity Analyzer")

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

data = ['3x3 team details final',
 'abu dhabi city run ',
 'abu dhabi municipality cycle race ',
 'abu dhabi tour cycle race ',
 'actvet run with gender',
 'adcb city run ',
 'adcb make a wish',
 'adcb pink run',
 'adcb run ',
 'adcb run race ',
 'adcb zayed sport city ',
 'adcb zayed sport city run data',
 'adcb zsc ',
 'adcc monday night cycle race ',
 'adnoc abu dhabi marathon ',
 'adnoc marathon ',
 'adnoc sunset run',
 'adsc triathlon',
 'al ain cycle race ',
 'al ain mounting bike challege',
 'al ain mtb challenge',
 'al ain night run',
 'al ain swim data',
 'al ain zoo ',
 'al ain zoo run ',
 'al hudayiryat duathlon',
 'al hudayriat ',
 'al hudayriat cycle race ',
 'al hudayriat dathlon',
 'al hudayriat island open water swim data',
 'al hudayriyat cycling race',
 'al hudrayriat triathlon ',
 'al mirfa cycling competition ',
 'al mirfa run',
 'al mirfa running competition',
 'al wathaba cycle race ',
 'al wathba cross country ',
 'al wathba cycle race ',
 'al wathba duathlon ',
 'al wathba hill climb ',
 'al wathba sportive ',
 'al wathba time trial ',
 'city run ',
 'cycle ',
 'cycle at train yas ',
 'cycle race yas ',
 'daman duathlon',
 'daman sunset cycle race',
 'daman trithlon ',
 'daman triyas',
 'das run ',
 'discover america run ',
 'dot open water swim',
 'du yas ',
 'duathlon ',
 'duathlon race ',
 'falcon daman challege race ',
 'fbma al ain zoo run',
 'fbma ladies run',
 'fbma ladies run data',
 'fbma run',
 'fbma run ',
 'five star ',
 'five start ',
 'global',
 'half marathon ',
 'hudayriyat cycling challenge',
 'hudayriyat duathlon',
 'individual time trial ',
 'indoor run ',
 'itt',
 'j triathlon and aquathlon',
 'junior triathlon and open aquathlon ',
 'junior triathlon and open aquathlon  al hudayriat island ',
 'junior triathlon and open aqwa',
 'knock out ',
 'liwa cycle ',
 'liwa cycle race data',
 'liwa urban ',
 'liwa urban sportive ',
 'maan at your own pace run',
 'monday night ',
 'night',
 'open aquathlon and junior triathlon',
 'open water swim ',
 'open water swim corniche beach',
 'race pogacar',
 'ras al khaimah hilfmarathon',
 'run',
 'run for smiles',
 'run the dock ',
 'senaat cycle ',
 'special olympics ',
 'stadium to stadium ',
 'stadium to stadium run',
 'stadium to stadium run ',
 'strider half marathon',
 'striders half maratho ',
 'sunset cycle race al wathba',
 'swim for clean sea ',
 'swimming ',
 'th fbma virtual ladies run',
 'the al wathba duathlon ',
 'the burjeel run',
 'time trial ',
 'triathlon ',
 'triventure',
 'uae tour al qudra ',
 'uae tour challenge jebel hafeet data',
 'uae tour liwa ',
 'uae tour mubadala challenge',
 'ultimate cross country championship',
 'westin kilomarathon ',
 'yas island cycle race ',
 'yas island ramdan night cycle race ',
 'yas island ramdan night run ',
 'year of tolerance run data',
 'year of tolerance swim data',
 'zoo run ']

preprocessed_data = [preprocess(item) for item in data]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_data)

cosine_similarities = cosine_similarity(tfidf_matrix)

similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.6)

filtered_cosine_similarities = np.where(cosine_similarities >= similarity_threshold, cosine_similarities, np.nan)

df = pd.DataFrame(filtered_cosine_similarities, columns=data, index=data)

fig = px.imshow(df)
fig.update_layout(
    title=f'Similarities between Events (>= {similarity_threshold * 100:.2f}%)',
    xaxis_title='Events',
    yaxis_title='Events',
    xaxis=dict(tickangle=45, tickfont=dict(size=8)),
    yaxis=dict(tickfont=dict(size=8)),
    height=1000,
    width=1000,
    margin=dict(l=200, r=50, t=100, b=200)
)

st.plotly_chart(fig)

similar_events = {event: [similar_event for similar_event, similarity in row.items() if similarity >= similarity_threshold and event != similar_event] for event, row in df.iterrows()}
filtered_similar_events = {event: similarities for event, similarities in similar_events.items() if similarities}
filtered_similar_events_df = pd.DataFrame(filtered_similar_events.items(), columns=['Event', 'Similar_Events'])

st.dataframe(filtered_similar_events_df, width=1500, height=500)

