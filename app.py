import streamlit as st
import pickle
import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import seaborn as sns


# ---------------------------
# Load ML Model
# ---------------------------
with open("sntmntanlys_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Database Connection
# ---------------------------
connection = mysql.connector.connect(
    host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    port=4000,
    user="2ZsEqVVabLFWFUp.root",
    password="16l1x13dLBPp8jug",
    database="sntmntanlys",
)
mycursor = connection.cursor()

# ---------------------------
# Streamlit App Config
# ---------------------------
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.title("📊 Sentiment Analysis Dashboard")

# Sidebar
st.sidebar.title("Choose an option:")
option = st.sidebar.radio(
    "Select:",
    ("Predict Sentiment", "EDA Insights", "Sentiment Analysis")
)

# ---------------------------
# Option 1: Predict Sentiment
# ---------------------------
if option == "Predict Sentiment":
    st.subheader("🔮 Predict Sentiment")
    user_input = st.text_area("Enter your review:")

    if st.button("Predict"):
        if user_input.strip():
            prediction = model.predict([user_input])[0]
            sentiment = "Positive" if prediction == 1 else ("Neutral" if prediction == 3 else "Negative")
            st.success(f"Predicted Sentiment: **{sentiment}**")
        else:
            st.warning("Please enter a review before predicting.")

# ---------------------------
# Option 2: EDA Insights
# ---------------------------
elif option == "EDA Insights":
    st.subheader("📊 Exploratory Data Analysis Insights")

    insight = st.selectbox(
        "Select an insight:",
        (
            "Distribution of Review Ratings",
            "Helpful Reviews (Threshold > 10)",
            "Common Keywords in Positive vs Negative Reviews",
            "Average Rating Over Time",
            "Ratings by User Location",
            "Platform-wise Average Ratings",
            "Verified vs Non-Verified Users",
            "Review Length by Rating",
            "Top Words in 1-Star Reviews",
            "Highest Rated ChatGPT Version"
        )
    )

    # ---- Insight 1 ----
    if insight == "Distribution of Review Ratings":
        mycursor.execute("""
            SELECT rating, COUNT(*) AS count
            FROM table1
            GROUP BY rating
            ORDER BY rating;
        """)
        result = mycursor.fetchall()
        df_ratings = pd.DataFrame(result, columns=["Rating", "Count"])

        st.write("### Ratings Distribution Table")
        st.dataframe(df_ratings)

        st.write("### Ratings Distribution Chart")
        fig, ax = plt.subplots()
        colors = plt.cm.Set3(range(len(df_ratings)))
        ax.bar(df_ratings["Rating"], df_ratings["Count"], color=colors, edgecolor="black")
        ax.set_xlabel("Rating (1 = worst, 5 = best)")
        ax.set_ylabel("Number of Reviews")
        ax.set_title("Distribution of Review Ratings")
        st.pyplot(fig)

    # ---- Insight 2 ----
    elif insight == "Helpful Reviews (Threshold > 10)":
        mycursor.execute("""
            SELECT 
                CASE WHEN helpful_votes > 10 THEN 'Helpful (>10)' ELSE 'Not Helpful (≤10)' END AS helpful_category,
                COUNT(*) AS count
            FROM table1
            GROUP BY helpful_category;
        """)
        result = mycursor.fetchall()
        df_helpful = pd.DataFrame(result, columns=["Helpful_Category", "Count"])

        st.write("### Helpful Reviews Table")
        st.dataframe(df_helpful)

        st.write("### Helpful Reviews Pie Chart")
        fig, ax = plt.subplots()
        ax.pie(
            df_helpful["Count"],
            labels=df_helpful["Helpful_Category"],
            autopct="%1.1f%%",
            colors=plt.cm.Pastel1.colors
        )
        ax.set_title("Helpful vs Not Helpful Reviews")
        st.pyplot(fig)

    # ---- Insight 3 ----
    elif insight == "Common Keywords in Positive vs Negative Reviews":
        mycursor.execute("SELECT review FROM table1 WHERE rating >= 4;")
        positive_reviews = " ".join([row[0] for row in mycursor.fetchall()])

        mycursor.execute("SELECT review FROM table1 WHERE rating <= 2;")
        negative_reviews = " ".join([row[0] for row in mycursor.fetchall()])

        def clean_text(text):
            text = re.sub(r"[^a-zA-Z\s]", "", text)
            return text.lower()

        positive_reviews = clean_text(positive_reviews)
        negative_reviews = clean_text(negative_reviews)

        st.write("### Word Cloud: Positive Reviews (4–5 Stars)")
        wc_pos = WordCloud(width=600, height=400, background_color="white", colormap="Greens").generate(positive_reviews)
        fig_pos, ax = plt.subplots()
        ax.imshow(wc_pos, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_pos)

        st.write("### Word Cloud: Negative Reviews (1–2 Stars)")
        wc_neg = WordCloud(width=600, height=400, background_color="white", colormap="Reds").generate(negative_reviews)
        fig_neg, ax = plt.subplots()
        ax.imshow(wc_neg, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_neg)

    # ---- Insight 4 ----
    elif insight == "Average Rating Over Time":
        mycursor.execute("""
            SELECT DATE(date) AS review_date, AVG(rating) AS avg_rating
            FROM table1
            GROUP BY review_date
            ORDER BY review_date;
        """)
        result = mycursor.fetchall()
        df_time = pd.DataFrame(result, columns=["Date", "Average_Rating"])
        df_time["Date"] = pd.to_datetime(df_time["Date"])

        st.write("### Average Rating Over Time (Table)")
        st.dataframe(df_time)

        st.write("### Average Rating Over Time (Line Chart)")
        fig, ax = plt.subplots()
        ax.plot(df_time["Date"], df_time["Average_Rating"], marker="o", color="blue", linestyle="-")
        ax.set_xlabel("Date")
        ax.set_ylabel("Average Rating")
        ax.set_title("Average Rating Trend Over Time")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ---- Insight 5 ----
    elif insight == "Ratings by User Location":
        mycursor.execute("""
            SELECT location, AVG(rating) AS avg_rating
            FROM table1
            GROUP BY location
            ORDER BY avg_rating DESC;
        """)
        result = mycursor.fetchall()
        df_location = pd.DataFrame(result, columns=["Location", "Average_Rating"])

        st.write("### Average Rating by Location Table")
        st.dataframe(df_location)

        st.write("### Average Rating by Location (World Map)")
        # Choropleth Map
        fig = px.choropleth(
            df_location,
            locations="Location",
            locationmode="country names",
            color="Average_Rating",
            color_continuous_scale="Viridis",
            title="Average Rating by User Location"
        )
        st.plotly_chart(fig)

    # ---- Insight 6 ----
    elif insight == "Platform-wise Average Ratings":
        mycursor.execute("""
            SELECT platform, AVG(rating) AS avg_rating
            FROM table1
            GROUP BY platform;
        """)
        result = mycursor.fetchall()
        df_platform = pd.DataFrame(result, columns=["Platform", "Average_Rating"])

        st.write("### Average Rating by Platform Table")
        st.dataframe(df_platform)

        st.write("### Average Rating by Platform (Bar Chart)")
        fig, ax = plt.subplots()
        colors = ["skyblue", "orange"]
        ax.bar(df_platform["Platform"], df_platform["Average_Rating"], color=colors, edgecolor="black")
        ax.set_xlabel("Platform")
        ax.set_ylabel("Average Rating")
        ax.set_title("Average Rating by Platform (Web vs Mobile)")
        st.pyplot(fig)

    # ---- Insight 7 ----
    elif insight == "Verified vs Non-Verified Users":
        mycursor.execute("""
            SELECT verified_purchase, AVG(rating) AS avg_rating
            FROM table1
            GROUP BY verified_purchase;
        """)
        result = mycursor.fetchall()
        df_verified = pd.DataFrame(result, columns=["Verified", "Average_Rating"])

        st.write("### Average Rating by Verified Purchase Table")
        st.dataframe(df_verified)

        st.write("### Verified vs Non-Verified Users (Bar Chart)")
        fig, ax = plt.subplots()
        colors = ["green", "red"]
        ax.bar(df_verified["Verified"], df_verified["Average_Rating"], color=colors, edgecolor="black")
        ax.set_xlabel("Verified Purchase")
        ax.set_ylabel("Average Rating")
        ax.set_title("Average Rating: Verified vs Non-Verified Users")
        st.pyplot(fig)
    # ---- Insight 8 ----
    elif insight == "Review Length by Rating":
        mycursor.execute("""
            SELECT rating, review_length
            FROM table1;
        """)
        result = mycursor.fetchall()
        df_length = pd.DataFrame(result, columns=["Rating", "Review_Length"])

        st.write("### Review Length by Rating Table")
        st.dataframe(df_length)

        st.write("### Review Length Distribution by Rating (Box Plot)")
        fig, ax = plt.subplots()
        df_length.boxplot(column="Review_Length", by="Rating", ax=ax, grid=False, patch_artist=True,
                        boxprops=dict(facecolor='skyblue', color='black'),
                        medianprops=dict(color='red'))
        ax.set_title("Review Length by Rating")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Review Length (Number of Characters)")
        plt.suptitle("")  # remove automatic "Boxplot grouped by group_by_column" title
        st.pyplot(fig)

    # ---- Insight 9 ----
    elif insight == "Top Words in 1-Star Reviews":
        mycursor.execute("""
            SELECT review
            FROM table1
            WHERE rating = 1;
        """)
        result = mycursor.fetchall()
        df_1star = pd.DataFrame(result, columns=["Review"])

        # Combine all reviews into one text
        text_1star = " ".join(df_1star["Review"].tolist())

        # Generate Word Cloud
        wordcloud = WordCloud(width=800, height=400, background_color="white",
                            colormap="Reds", stopwords=set(STOPWORDS)).generate(text_1star)

        st.write("### Word Cloud: Top Words in 1-Star Reviews")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    # ---- Insight 10 ----
    elif insight == "Highest Rated ChatGPT Version":
        mycursor.execute("""
            SELECT version, AVG(rating) AS avg_rating
            FROM table1
            GROUP BY version
            ORDER BY avg_rating DESC;
        """)
        result = mycursor.fetchall()
        df_version = pd.DataFrame(result, columns=["Version", "Average_Rating"])

        st.write("### Average Rating by ChatGPT Version Table")
        st.dataframe(df_version)

        st.write("### Average Rating by ChatGPT Version (Bar Chart)")
        fig, ax = plt.subplots(figsize=(8,5))
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_version)))  # different color for each bar
        ax.bar(df_version["Version"], df_version["Average_Rating"], color=colors, edgecolor="black")
        ax.set_xlabel("ChatGPT Version")
        ax.set_ylabel("Average Rating")
        ax.set_title("Average Rating by ChatGPT Version")
        plt.xticks(rotation=45)
        st.pyplot(fig)    
# ---------------------------
# Option 3: Sentiment Analysis
# ---------------------------
elif option == "Sentiment Analysis":
    st.subheader("📈 Sentiment Analysis Insights")

    insight = st.selectbox(
        "Select a sentiment insight:",
        ("Overall Sentiment of Reviews",
         "Sentiment vs Rating",
         "Keywords per Sentiment",
         "Sentiment Trend Over Time",
         "Verified Users vs Non-Verified Sentiment",
         "Review Length vs Sentiment",
         "Sentiment by Location",
         "Sentiment by Platform",
         "Sentiment by Version",
         "Negative Feedback Themes"
         )
    )

    # Load pipeline once
    with open('sntmntanlys_pipeline.pkl', 'rb') as file:
        sntmntanlys_pipeline = pickle.load(file)

    # Fetch reviews and ratings from DB
    mycursor.execute("SELECT date, review, rating, verified_purchase rating FROM table1;")
    data = mycursor.fetchall()
    df = pd.DataFrame(data, columns=['date', 'review', 'rating', 'verified_purchase'])

    # Ensure reviews are strings
    df['review'] = df['review'].astype(str)
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    if insight == "Overall Sentiment of Reviews":
        # Predict sentiment
        df['Sentiment'] = df['review'].apply(lambda x: sntmntanlys_pipeline.predict([x])[0])

        # Aggregate counts
        sentiment_counts = df['Sentiment'].value_counts()
        df_sentiment = pd.DataFrame({
            'Sentiment': sentiment_counts.index,
            'Count': sentiment_counts.values
        })

        st.write("### Overall Sentiment Table")
        st.dataframe(df_sentiment)

        # Pie chart
        fig, ax = plt.subplots(figsize=(6,6))
        sizes = df_sentiment['Count']
        labels = df_sentiment['Sentiment']
        colors = ['#66b3ff', '#ffcc99', '#ff6666']  # Positive, Neutral, Negative
        explode = [0.05]*len(labels)

        ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            explode=explode,
            shadow=True
        )
        ax.axis('equal')  # Equal aspect ratio ensures pie is circular
        st.pyplot(fig)

    elif insight == "Sentiment vs Rating":
        # Predict sentiment
        df['Sentiment'] = df['review'].apply(lambda x: sntmntanlys_pipeline.predict([x])[0])

        # Aggregate counts
        df_grouped = df.groupby(['rating', 'Sentiment']).size().reset_index(name='Count')

        # Grouped bar chart
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(
            data=df_grouped,
            x='rating',
            y='Count',
            hue='Sentiment',
            palette=['#66b3ff','#ffcc99','#ff6666'],
            ax=ax
        )
        ax.set_title("Sentiment Distribution per Rating")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Number of Reviews")
        st.pyplot(fig)

    elif insight == "Keywords per Sentiment":
        from wordcloud import WordCloud, STOPWORDS

        # Predict sentiment
        df['Sentiment'] = df['review'].apply(lambda x: sntmntanlys_pipeline.predict([x])[0])

        # Generate word clouds for each sentiment
        sentiments = df['Sentiment'].unique()
        for sentiment in sentiments:
            text = " ".join(df[df['Sentiment'] == sentiment]['review'].tolist())

            if text.strip():  # Avoid empty string errors
                wc = WordCloud(
                    width=800,
                    height=400,
                    background_color="white",
                    colormap="Set2",
                    stopwords=set(STOPWORDS)
                ).generate(text)

                st.write(f"### Word Cloud for {sentiment} Reviews")
                fig, ax = plt.subplots(figsize=(8,4))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.write(f"No text available for {sentiment} sentiment.")

    elif insight == "Sentiment Trend Over Time":
        # Predict sentiment
        df['Sentiment'] = df['review'].apply(lambda x: sntmntanlys_pipeline.predict([x])[0])

        # Group by month-year
        df['Month'] = df['date'].dt.to_period('M').astype(str)
        trend_df = df.groupby(['Month', 'Sentiment']).size().reset_index(name='Count')

        # Pivot for stacked area chart
        pivot_df = trend_df.pivot(index='Month', columns='Sentiment', values='Count').fillna(0)

        st.write("### Sentiment Trend Over Time (Monthly)")
        fig, ax = plt.subplots(figsize=(10,6))
        pivot_df.plot(kind='area', stacked=True, ax=ax,
                      cmap='coolwarm', alpha=0.7)
        ax.set_title("Sentiment Trend Over Time")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Reviews")
        st.pyplot(fig)

    elif insight == "Verified Users vs Non-Verified Sentiment":
        # Predict sentiment
        df['Sentiment'] = df['review'].apply(lambda x: sntmntanlys_pipeline.predict([x])[0])

        # Group by verified_purchase & sentiment
        df_grouped = df.groupby(['verified_purchase', 'Sentiment']).size().reset_index(name='Count')

        # Pivot for stacked bar chart
        pivot_df = df_grouped.pivot(index='verified_purchase', columns='Sentiment', values='Count').fillna(0)

        st.write("### Sentiment Distribution: Verified vs Non-Verified Users")
        st.dataframe(df_grouped)

        # Plot stacked bar chart
        fig, ax = plt.subplots(figsize=(8,6))
        pivot_df.plot(kind='bar', stacked=True, ax=ax,
                      color=['#66b3ff','#ffcc99','#ff6666'], alpha=0.85)
        ax.set_title("Verified vs Non-Verified Sentiment Distribution")
        ax.set_xlabel("Verified Purchase")
        ax.set_ylabel("Number of Reviews")
        ax.legend(title="Sentiment")
        st.pyplot(fig)

    elif insight == "Review Length vs Sentiment":
        # Fetch reviews + review_length
        mycursor.execute("SELECT review, review_length FROM table1;")
        data = mycursor.fetchall()
        df = pd.DataFrame(data, columns=['review', 'review_length'])
        df['review'] = df['review'].astype(str)

        # Predict sentiment
        df['Sentiment'] = df['review'].apply(lambda x: sntmntanlys_pipeline.predict([x])[0])

        # Show table for quick reference
        st.write("### Review Length vs Sentiment Table")
        st.dataframe(df[['review_length', 'Sentiment']].groupby('Sentiment').mean().reset_index())

        # Box plot visualization
        fig, ax = plt.subplots(figsize=(8,6))
        sns.boxplot(
            data=df,
            x='Sentiment',
            y='review_length',
            palette=['#66b3ff','#ffcc99','#ff6666'],
            ax=ax
        )
        ax.set_title("Review Length Distribution per Sentiment")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Review Length (characters or words)")
        st.pyplot(fig)

    elif insight == "Sentiment by Location":
        # Fetch reviews + locations
        mycursor.execute("SELECT review, location FROM table1;")
        data = mycursor.fetchall()
        df = pd.DataFrame(data, columns=['review', 'location'])
        df['review'] = df['review'].astype(str)

        # Predict sentiment
        df['Sentiment'] = df['review'].apply(lambda x: sntmntanlys_pipeline.predict([x])[0])

        # Aggregate counts
        sentiment_counts = df.groupby(['location', 'Sentiment']).size().reset_index(name='Count')
        st.write("### Sentiment Distribution by Location (Table)")
        st.dataframe(sentiment_counts)

        # Pivot for easier mapping
        sentiment_pivot = sentiment_counts.pivot(index='location', columns='Sentiment', values='Count').fillna(0).reset_index()

        # For visualization, we’ll compute % positive per location
        sentiment_pivot['Total'] = sentiment_pivot.sum(axis=1, numeric_only=True)
        if 'Positive' in sentiment_pivot.columns:
            sentiment_pivot['Positive %'] = (sentiment_pivot['Positive'] / sentiment_pivot['Total']) * 100
        else:
            sentiment_pivot['Positive %'] = 0

        import plotly.express as px
        st.write("### World Map of Positive Sentiment % by Location")

        fig = px.choropleth(
            sentiment_pivot,
            locations="location",
            locationmode="country names",  # assumes country names in 'location'
            color="Positive %",
            hover_name="location",
            color_continuous_scale="Blues",
            title="Positive Sentiment (%) by Location"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif insight == "Sentiment by Platform":
        # Fetch reviews + platform
        mycursor.execute("SELECT review, platform FROM table1;")
        data = mycursor.fetchall()
        df = pd.DataFrame(data, columns=['review', 'platform'])
        df['review'] = df['review'].astype(str)

        # Predict sentiment
        df['Sentiment'] = df['review'].apply(lambda x: sntmntanlys_pipeline.predict([x])[0])

        # Aggregate counts
        df_grouped = df.groupby(['platform', 'Sentiment']).size().reset_index(name='Count')
        st.write("### Sentiment Distribution by Platform (Table)")
        st.dataframe(df_grouped)

        # Grouped bar chart
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(
            data=df_grouped,
            x='platform',
            y='Count',
            hue='Sentiment',
            palette=['#66b3ff','#ffcc99','#ff6666'],
            ax=ax
        )
        ax.set_title("Sentiment Comparison: Web vs Mobile")
        ax.set_xlabel("Platform")
        ax.set_ylabel("Number of Reviews")
        st.pyplot(fig)

    elif insight == "Sentiment by Version":
        # Fetch reviews + version
        mycursor.execute("SELECT review, version FROM table1;")
        data = mycursor.fetchall()
        df = pd.DataFrame(data, columns=['review', 'version'])
        df['review'] = df['review'].astype(str)

        # Predict sentiment
        df['Sentiment'] = df['review'].apply(lambda x: sntmntanlys_pipeline.predict([x])[0])

        # Aggregate counts
        df_grouped = df.groupby(['version', 'Sentiment']).size().reset_index(name='Count')
        st.write("### Sentiment Distribution by Version (Table)")
        st.dataframe(df_grouped)

        # Pivot for heatmap
        pivot_df = df_grouped.pivot(index='version', columns='Sentiment', values='Count').fillna(0)

        # Heatmap visualization
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt="g",
            cmap="YlGnBu",
            cbar=True,
            linewidths=0.5,
            linecolor="gray",
            ax=ax
        )
        ax.set_title("Sentiment Distribution by ChatGPT Version")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Version")
        st.pyplot(fig)

    elif insight == "Negative Feedback Themes":
        # Fetch reviews
        mycursor.execute("SELECT review FROM table1;")
        reviews = [row[0] for row in mycursor.fetchall()]
        df = pd.DataFrame(reviews, columns=['review'])
        df['review'] = df['review'].astype(str)

        # Predict sentiment
        df['Sentiment'] = df['review'].apply(lambda x: sntmntanlys_pipeline.predict([x])[0])

        # Filter only negative reviews
        negative_reviews = df[df['Sentiment'] == "Negative"]['review'].tolist()

        if len(negative_reviews) > 5:  # need enough data for topic modeling
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation

            # Vectorize text
            vectorizer = CountVectorizer(
                max_df=0.9, min_df=2, stop_words='english'
            )
            X = vectorizer.fit_transform(negative_reviews)

            # LDA topic modeling
            lda = LatentDirichletAllocation(n_components=5, random_state=42)
            lda.fit(X)

            # Extract top keywords per topic
            terms = vectorizer.get_feature_names_out()
            topics = []
            for idx, topic in enumerate(lda.components_):
                top_terms = [terms[i] for i in topic.argsort()[-10:]]
                topics.append({"Topic": f"Theme {idx+1}", "Keywords": ", ".join(top_terms)})

            topics_df = pd.DataFrame(topics)

            st.write("### Top Negative Feedback Themes (Topic Modeling)")
            st.dataframe(topics_df)

            # Count keyword frequencies (top 15 words overall)
            word_counts = X.toarray().sum(axis=0)
            word_freq = pd.DataFrame({
                "Word": terms,
                "Frequency": word_counts
            }).sort_values(by="Frequency", ascending=False).head(15)

            # Plot keyword bar chart
            fig, ax = plt.subplots(figsize=(10,6))
            sns.barplot(
                data=word_freq,
                y="Word",
                x="Frequency",
                palette="Reds_r",
                ax=ax
            )
            ax.set_title("Most Common Negative Feedback Keywords")
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Keyword")
            st.pyplot(fig)

        else:
            st.warning("Not enough negative reviews available for topic modeling.")

    