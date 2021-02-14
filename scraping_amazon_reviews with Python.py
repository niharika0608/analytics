#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install lxml')


# In[2]:


get_ipython().system('pip install requests')


# In[4]:


import pandas as pd


# In[5]:


import requests #make http calls


# In[6]:


from lxml import html #Parsing the html


# In[8]:


amazon_url_to_crawl = "https://www.amazon.in/Test-Exclusive-558/product-reviews/B077PWJRFH/?pageNumber=2"


# In[14]:


user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36"


# In[16]:


headers = {'User-Agent': user_agent}


# In[17]:


page = requests.get(amazon_url_to_crawl, headers=headers)


# In[18]:


page


# In[19]:


page.content


# In[20]:


#specifying the parser
parser = html.fromstring(page.content)


# In[21]:


parser


# In[22]:


xpath_reviews = '//div[@data-hook="review"]'
reviews = parser.xpath(xpath_reviews)
reviews


# In[31]:


for review in reviews: 
    user_name = review.xpath(".//div[@class='a-profile-content']//text()")[0]
    star_rating = review.xpath(".//i[@data-hook='review-star-rating']//text()")[0]
    print(user_name)


# In[37]:


for review in reviews: 
    user_name = review.xpath(".//div[@class='a-profile-content']//text()")[0]
    star_rating = review.xpath(".//i[@data-hook='review-star-rating']//text()")[0]
    body = review.xpath(".//span[@data-hook='review-body']//text()")[0]
    print(body)


# In[39]:


#to find reveiw dates
for review in reviews: 
    user_name = review.xpath(".//div[@class='a-profile-content']//text()")[0]
    star_rating = review.xpath(".//i[@data-hook='review-star-rating']//text()")[0]
    body = review.xpath(".//span[@data-hook='review-body']//text()")
    date = review.xpath(".//span[@data-hook='review-date']//text()")[0]
    print(date)


# In[38]:


#can also split the element to view only dates
# if used 0 gives actual date and if 1 it will give first element
for review in reviews:
    user_name = review.xpath(".//div[@class='a-profile-content']//text()")[0]
    star_rating = review.xpath(".//i[@data-hook='review-star-rating']//text()")[0]
    body = review.xpath(".//span[@data-hook='review-body']//text()")
    date = review.xpath(".//span[@data-hook='review-date']//text()")[0].split("on ")[1]
    print(date)


# In[40]:


for review in reviews:
    user_name = review.xpath(".//div[@class='a-profile-content']//text()")[0]
    star_rating = review.xpath(".//i[@data-hook='review-star-rating']//text()")[0]
    body = review.xpath(".//span[@data-hook='review-body']//text()")
    date = review.xpath(".//span[@data-hook='review-date']//text()")[0].split("on ")[0]
    print(date)


# In[41]:


reviews_dataframe = pd.DataFrame()
for review in reviews:
    user_name = review.xpath(".//div[@class='a-profile-content']//text()")[0]
    star_rating = review.xpath(".//i[@data-hook='review-star-rating']//text()")[0]
    body = review.xpath(".//span[@data-hook='review-body']//text()")
    date = review.xpath(".//span[@data-hook='review-date']//text()")[0].split("on ")[1]
    reviews_dict = {
        'user_name': user_name,
        'star_rating': star_rating,
        'body': body,
        'date': date
    }
    reviews_dataframe = reviews_dataframe.append(reviews_dict, ignore_index=True)
    
reviews_dataframe.to_csv("amazon_product_reviews.csv")


# In[ ]:




