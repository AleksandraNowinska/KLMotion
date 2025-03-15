# ✈ SkySentiment: Analyzing Airline Twitter Sentiment

## 📌 Project Overview
**SkySentiment** is a data-driven project analyzing customer sentiment on Twitter in response to interactions with major airlines. The goal is to investigate **how response times impact sentiment evolution** and to provide actionable insights for improving airline social media engagement.

We extracted **Twitter conversations**, applied **natural language processing (NLP) techniques**, and used **statistical modeling** to understand sentiment shifts across different airlines.

---

## 🔍 Key Features
✔ **Twitter Data Processing**: Extracted, cleaned, and structured conversations from MongoDB.  
✔ **Sentiment Analysis**: Used VADER and text preprocessing (lemmatization, translation) to assess sentiment evolution.  
✔ **Response Time Impact**: Analyzed how customer sentiment changes depending on airline response speed.  
✔ **Regression Modeling**: Evaluated sentiment shifts using **OLS regression** and visualization techniques.  
✔ **Comparative Analysis**: Compared KLM with major competitors like British Airways, Lufthansa, and Air France.  

---

## 🛠 Tech Stack
- **Python**: Pandas, NumPy, Matplotlib, Seaborn, Statsmodels, Scikit-learn  
- **MongoDB**: Stored and queried large-scale Twitter data  
- **Natural Language Processing**: VADER Sentiment Analysis, Text Lemmatization, Translation  
- **Machine Learning**: Regression models for predicting sentiment evolution  
- **Visualization**: Box plots, scatter plots, and line graphs for trend analysis  

---

## 🚀 Installation & Setup

1️⃣ **Clone the Repository**  
git clone https://github.com/yourusername/SkySentiment.git
cd SkySentiment
   
2. Install dependencies:
   pip install -r requirements.txt
   
3. Set Up MongoDB (Optional: If processing raw tweets)

a. Ensure MongoDB is installed and running
mongod --dbpath /path/to/mongodb

b. Load data into MongoDB (if needed):
python data_loading.py

4. Run Sentiment Analysis
python sentiment_evolution.py

5. Train Regression Model
python business_idea.py

   
## 📊 Results & Insights
Faster response times do not always guarantee better sentiment.
Engagement matters: The number of tweets in a conversation influences sentiment evolution more than speed.
Some competitors achieve higher sentiment improvement despite slower responses.
The model suggests optimizing customer engagement strategies beyond just fast replies.

## 📌 Next Steps
🔹 Incorporate Real-Time Sentiment Tracking
🔹 Experiment with Deep Learning Models (e.g., LSTMs, Transformers)
🔹 Extend Analysis to Additional Airlines and Languages

## 🤝 Contributing
Want to improve the project? Feel free to fork, submit pull requests, or discuss ideas in Issues!
