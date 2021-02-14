library(twitteR)
library(ROAuth)
library(plyr)
library(dplyr)
library(stringr)
library(ggplot2)
library(httr)
library(wordcloud)
library(RCurl)
library(syuzhet)
library(tm)
library(tidytext)


#OAuth
oauth_endpoint(authorize = "https://api.twitter.com/oauth",
               access = "https://api.twitter.com/oauth/access_token")

#these given url's will not be useful anywhere apart from twitter
#connect to API
reqURL <- 'https://api.twitter.com/oauth/request_token'
accessURL <- 'https://api.twitter.com/oauth/access_token'
authURL <- 'https://api.twitter.com/oauth/authorize'

#Twitter Application
consumerKey="" #Replace with your consumerKey
consumerSecret="" #Replace with your consumerSecret
accesstoken="" #Replace with your accesstoken
accesssecret="" #Replace with your accesssecret

setup_twitter_oauth(consumer_key=consumerKey, consumer_secret=consumerSecret, access_token =accesstoken, access_secret = accesssecret)

trend_locations <- availableTrendLocations()
View(trend_locations)

#to find hot topics or trending topics at particular location
#we need to find out first WOEID of the location

city_woeid = subset(trend_locations, name == "Toronto")$woeid
city_woeid

toronto_city_woeid = subset(trend_locations, name == "Toronto")$woeid
trending_topics_toronto = getTrends(woeid=toronto_city_woeid)
View(trending_topics_toronto)

city_woeid = subset(trend_locations, name == "Bangalore")$woeid
trending_topics = getTrends(woeid=city_woeid)
View(trending_topics)

city_woeid = subset(trend_locations, name == "India")$woeid
trending_topics = getTrends(woeid=city_woeid)
View(trending_topics)

#21/01/2021
#if we are not specifying until and since in the statement 
#than number of tweets will be random
#to pass the language for not to get mixed language text use 'en' code for english only
tweets =  searchTwitter("#mumbai", n=5000, lang = 'en')
length(tweets)
tweets

#changing tweets into dataframe use ldply
tweets_dataframe <- ldply(tweets, function(t) t$toDataFrame())
View(tweets_dataframe)

write.csv(tweets_dataframe,"tweets.csv")

txt = sapply(tweets, function(x) x$getText())
View(txt)

txt = gsub("(RT|via)((?:\\b\\W*@\\w+)+)","", txt)
txt = gsub("http[^[:blank:]]+", "", txt)
txt = gsub("@\\w+", "", txt)
txt = gsub("[[:punct:]]", " ", txt)

#Remove alphanumeric
txt = gsub("[^[:alnum:]]", " ", txt)

#22/01/2021
# Create Corpus
txt = VCorpus(VectorSource(txt))

# Convert to lower case
txt = tm_map(txt, content_transformer(tolower))

#remove all stop words
txt = tm_map(txt, removeWords, stopwords("english"))

#remove all white spaces
txt = tm_map(txt, stripWhitespace)

txt[[1]]$content

# Step 4 --> Data modeling
pal <- brewer.pal(8,"Dark2")
wordcloud(txt, min.freq = 15,  max.words = 200, width=4000, height =14000,  random.order = FALSE, color=pal)

pal <- brewer.pal(8,"Accent")
wordcloud(txt, min.freq = 15,  max.words = 200, width=1000, height =1000,  random.order = FALSE, color=pal)

?get_nrc_sentiment
sent = "It is going to rain heavily today so I may not be able play cricket"
sentime<- get_nrc_sentiment(sent)
View(sentime)

mysentiment <- get_nrc_sentiment(as.character(txt))
SentimentScores <- data.frame(colSums(mysentiment[,]))
names(SentimentScores) <- "Score"
SentimentScores <- cbind("sentiment" = rownames(SentimentScores), SentimentScores)
rownames(SentimentScores) <- NULL
ggplot(data = SentimentScores, aes(x = sentiment, y = Score)) +
  geom_bar(aes(fill = sentiment), stat = "identity") +
  theme(legend.position = "none") +
  xlab("Sentiment") + ylab("Score") + ggtitle("Total Sentiment Score Based on Tweets")

head(sentiments)
get_sentiments("afinn")
View(get_sentiments("afinn"))
View(get_sentiments("bing"))
View(get_sentiments("nrc"))

#Assignment
#wordcloud
#sentiments
#Uber
#Ola

#Compare the telecom operators in India
#Airtel
#Vodafone
#Idea
#Jio
#BSNL

#Compare various food delivery services
#Swiggy
#Zomato

#Compare mobile phones
#Google
#Apple
#Samsung

#compare online flight services
#yatra.com
#cleartrip.com
#

#Prepare report on comparison should be in pdf so that charts can come

