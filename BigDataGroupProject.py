# Databricks notebook source
#Import libraries
from pyspark.sql.functions import col, sum, when, split, regexp_replace, explode, length, size, mean, desc
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
!pip install tqdm
from tqdm import tqdm
!pip install nltk
import nltk
from nltk.corpus import stopwords

# COMMAND ----------

# MAGIC %md Import dataset and analyse it
# MAGIC

# COMMAND ----------

# Load data into the big data cluster as a dataframe
df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/bernardo.machado4@protonmail.com/cryptonews.csv")
df.printSchema() 

# COMMAND ----------

#Print all the database
df.show()

# COMMAND ----------

#Find source with most entries
df.groupBy('source').count().orderBy('count', ascending = 0).show()

# COMMAND ----------

#Find most comun subject
df.groupBy('subject').count().orderBy('count', ascending = 0).show()

# COMMAND ----------

"""We see we have more data the the sentiment evaluation (polarity and subjectivity) but we decided to ingro them and just use the sentiment in our ML apporoach"""
df.select("sentiment").show(truncate=False)

# COMMAND ----------

#Create a new column with only the sentiment information
df = df.withColumn("OnlySentiment", split(df.sentiment, " ").getItem(1))
df.show()

# COMMAND ----------

#Clean the strings in cloumn OnlySentiment
df = df.withColumn("OnlySentiment", split(df.OnlySentiment, "'").getItem(1))
df.show()

# COMMAND ----------

#Analise of the general sentiment
df.groupBy('OnlySentiment').count().orderBy('count', ascending=0).show()

# COMMAND ----------

#Create a new column with integers to use in a machine learning model
df = df.withColumn("IntSentiment", when(df.OnlySentiment == "negative", 0)
                                        .when(df.OnlySentiment == "positive", 1)
                                        .otherwise(3))

df.show()

# COMMAND ----------

df.printSchema() #confirm data in IntSentiment is an integer

# COMMAND ----------

#Find the most comun sentiment in March 2023
df.where(col("date").between("2023-03-01", "2023-03-31")).show()

# COMMAND ----------

#Create a new column with a list of words from text column
df = df.withColumn("wordsDict", split(df.text, " "))
df.show()

# COMMAND ----------

#Count the number of words of each row for  text column
df = df.withColumn("wordCount", size(df["wordsDict"]))
df.show()

# COMMAND ----------

# MAGIC %md Apply RDDs

# COMMAND ----------

# Convert Dataset to RDD
rdd = df.rdd

# Extract the desired column
text_rdd = rdd.map(lambda row: row.text)

# Split lines into words
words_rdd = text_rdd.flatMap(lambda line: line.split(" "))

# Count the occurrences of each word
word_counts = words_rdd.countByValue()

# print the word count
for word, count in word_counts.items():
    print(f"{word}: {count}")

# COMMAND ----------

# Sort the word count in descending order
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# Print the word count in descending order
for word, count in sorted_word_counts:
    print(f"{word}: {count}")
    

# COMMAND ----------

#Remove punctuation to find the tru most comun words
def remove_punctuation(text):
    # Use regular expression to replace punctuation with empty string
    return re.sub(r'[^\w\s]', '', text)

clean_rdd = words_rdd.map(remove_punctuation).map(lambda x: x.lower())

# Count the occurrences of each word and sorte them in descending order
clean_word_counts = clean_rdd.countByValue()
sorted_clean_word_counts = sorted(clean_word_counts.items(), key=lambda x: x[1], reverse=True)

# Print the word count in descending order
for word, count in sorted_word_counts:
    print(f"{word}: {count}")

# COMMAND ----------

#Word count with RDD

# Convert DataFrame to RDD and calculate sum of words for each row
word_count_rdd = df.select("text").rdd.map(lambda row: (row[0], len(row[0].split(" "))))
text_list = list()
count_list = list()
# Print the sum of words for each row
for row in word_count_rdd.collect():
    print(f"Text: {row[0]} <---> Word Count: {row[1]}")
    text_list.append(row[0])
    count_list.append(row[1])

# COMMAND ----------

#Count all word of text column using an RDD
word_count_rdd.reduce(lambda x, y : ("Total", x[1]+y[1]))

# COMMAND ----------

# MAGIC %md Apply a machine learning model to predict sentiment of our news

# COMMAND ----------

#Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(df.text, df.IntSentiment, test_size=0.20, random_state=4)

# COMMAND ----------

nltk.download('stopwords')
stop = set(stopwords.words('english')) #set with stopwords for later remove from tect

def clean(text_list):
    
    updates = []
    
    for j in tqdm(text_list):
        
        text = j
        
        #LOWERCASE TEXT
        text = text.lower()
        
        #REMOVE NUMERICAL DATA and PUNCTUATION
        text = re.sub("[^a-zA-Z]"," ", text )
        
        #REMOVE STOPWORDS
        text = " ".join([word for word in text.split() if word not in stop])
        
        #Lemmatize
        text = " ".join(lemma.lemmatize(word) for word in text.split())
            
        updates.append(text)
        
    return updates
