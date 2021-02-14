#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install networkx


# In[2]:


import networkx as nx


# In[3]:


#Build a symmetric network

g_symmetric = nx.Graph() #initialized an empty network


# In[4]:


g_symmetric.add_edge('Amitabh', 'Dev')
g_symmetric.add_edge('Amitabh', 'Akshay')
g_symmetric.add_edge('Amitabh', 'Aamir')
g_symmetric.add_edge('Amitabh', 'Abhishek')
g_symmetric.add_edge('Abhishek', 'Aamir')
g_symmetric.add_edge('Abhishek', 'Dev')
g_symmetric.add_edge('Abhishek', 'Akshay')
g_symmetric.add_edge('Dev', 'Aamir')


# In[34]:


#matplotlib
#I want to see all the charts in the same notebook, I am working on<-by using inline

get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


get_ipython().magic("matplotlib inline")
nx.draw_networkx(g_symmetric)


# In[35]:


nx.degree(g_symmetric, "Akshay")


# In[33]:


nx.average_clustering(g_symmetric)


# In[7]:


g_asymmetric = nx.DiGraph() #empty network


# In[8]:


g_asymmetric.add_edge('A','B')
g_asymmetric.add_edge('A','D')
g_asymmetric.add_edge('C','A')
g_asymmetric.add_edge('D','E')


# In[9]:


nx.draw_networkx(g_asymmetric)


# In[10]:


#weighted graph

g_weighted = nx.Graph()
g_weighted.add_edge('Amitabh', 'Dev', weight = 4)
g_weighted.add_edge('Amitabh', 'Akshay', weight = 8)
g_weighted.add_edge('Amitabh', 'Aamir',weight = 12)
g_weighted.add_edge('Amitabh', 'Abhishek', weight = 2)
g_weighted.add_edge('Abhishek', 'Aamir', weight = 3)
g_weighted.add_edge('Abhishek', 'Dev', weight = 1)
g_weighted.add_edge('Abhishek', 'Akshay', weight = 9)
g_weighted.add_edge('Dev', 'Aamir', weight = 2)
nx.draw_networkx(g_weighted)


# In[11]:


#Betweenness Centrality<- Algo used for influencer analysis

influ = nx.betweenness_centrality(g_symmetric)


# In[12]:


#it will be in the form of dictionary
influ


# In[13]:


sorted(influ, key=influ.get)


# In[14]:


#influencers on the basis of value<- you can use lambda function in python
sorted(influ.items(), key=lambda x: x[1], reverse=True)


# In[15]:


#for only top two influencers
sorted(influ.items(), key=lambda x: x[1], reverse=True)[:2]


# In[16]:


fb_graph = nx.read_edgelist("C:/Users/Niharika/Desktop/MBA-BA-2nd sem/Social Media Analytics/facebook_combined.txt")


# In[17]:


nx.draw_networkx(fb_graph)


# In[18]:


nx.info(fb_graph)


# In[19]:


influ = nx.betweenness_centrality(fb_graph)


# In[20]:


influ


# In[21]:


sorted(influ.items(), key=lambda x: x[1], reverse=True)[:5]


# In[28]:


fb_graph = nx.read_edgelist("C:/Users/Niharika/Desktop/MBA-BA-2nd sem/Social Media Analytics/tableaubudget2021.txt")


# In[29]:


nx.info(fb_graph)


# In[30]:


influ = nx.betweenness_centrality(fb_graph)


# In[31]:


influ


# In[32]:


sorted(influ.items(), key=lambda x: x[1], reverse=True)[:5]


# In[ ]:




