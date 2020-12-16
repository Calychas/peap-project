This is template for project's repository, where all the code should be maintaned. 
If there is a need for large file storage please use external resources.

Please implement logical structure of the project to make it easy to understand for teachers.

Please fill all phases according to the project of your project proposal.

# Phase 1

Derivables:
- Preparation of list of users  
- Scrapping tweets of users (What is a limit for dates? From the beginning of user creation?)  
- Initial data processing - how to aggregate all tweets for one user? Which NLP model to use for the embeddings? 


What was achieved:
- **782** politicians were identified
- **581** tweeter account were found
- Out of those - **548** met requirements for minimal activity
- **1 659 884** tweets were gathered (all tweets since each account creation)


# Phase 2

Derivables:
- Creation of embeddings for each user.  
- Implementation of machine learning model (most possible - unsupervised) that will classify/cluster embeddings  
- In-depth analysis of results of classification/clustering  


What was achieved:
- Tweets advanced cleaning:
  - Hashtag sign and mentions removal
  - Links removed
  - Emoji to text
  - Lemmatization using KRNNT tagger
  - Stopwords removal
- Sentiment model creation:
  - Sample of tweets annotated
  - Sentiment classifier based on fastText and data from CLARIN-PL and annotated by us
  - Example emotional texts from PLWORDNET extracted
- Topic modelling:
  - Trained LDA model
  - Topics basic analysis for users/coallitions/...
- Embeddings for users created based on cleaned tweets using HerBERT
- Basic clustering analysis on tweets embeddings performed


# Phase 3

Derivables:
- Gathering all results and plots that were extracted during data analysis.  
- Implementation of website that includes results and plots  
- Deployment of website to public  
- Creation of poster

What was achieved:
- ...
