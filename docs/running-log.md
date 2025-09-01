# Running Diary

## Saturday
Got the FastAPI structure set up, pretty straight forward. Pulled in the OpenAI key and made some sample calls, pulled a set of the huggingface data locally.

Have:
- postgres w/ pgvector
- docker compose

Thinking through how to work with this data. First idea is to create some first text embeddings for the product that's a little beefed up and go from there. Thinking about
maybe having an up-front LLM call that takes an initial query and gets it into the general format that the embeddings are looking for? Or at least provides some guidance on interpreting the initial query.

Also do want to start with creating some test queries to make sure that this makes sense. 


Thinking through now which vector database to work with, think that there may be some opportunities to do some metadata filtering given what the dataset is looking like. Given I'm using docker for the build, I'm looking into Chroma v Milvus v PGvector. Looks like pgvector after 0.8.0 supports metadata filtering in the way that I'd like. With the dataset being so large I'm going to need to use just vector similarity rather than nearest neighbors.

Don't think I need to add any fields to the data for now, and I'm only saving 300 records. Will probably start by just creating the dataspec and loading at application launch.

I'm doing a couple things in parallel. Want to set up the appropriate tests module so that I can create some way to view different versions of searching through the data. I also want an easy docker compose for playing around with this in dev.

Now I have my data loader set up, I want to have it actually make a new table for every load so I can test loading different things in as vectors and evaluate performance against the queries. 

In parallel, working on the chat api to return streaming data as a start.

Adding in MLFlow so I can organize some of the experiments that I'll be running across the different sets of embeddings.

Took me a little longer than I would like to get that set up but think it should be worth the hassle at some point.

Also need to keep streaming out of this here for a little bit, which is also fine.