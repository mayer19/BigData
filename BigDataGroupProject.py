# Databricks notebook source
#Import libraries
from pyspark.sql.functions import mean
from pyspark.sql.functions import desc
from pyspark.sql.functions import col, sum, when, split, regexp_replace, explode, length, size

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

#Create a new column with only the sentiment information
df = df.withColumn("wordsDict", split(df.text, " "))
df.show()

# COMMAND ----------

df = df.withColumn("wordCount", size(df["wordsDict"]))
df.show()

# COMMAND ----------

# Convert DataFrame to RDD
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


