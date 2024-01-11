## Personal thoughts regarding the project:  
In my opinion, generating insights through summarization (e.g. extracting key information using a constant prompt for every reviews) is more appropriate as opposed to doing QnA on the reviews dataset. QnA via RAG is more appropriate when we need to search a specific information from a large corpus of documents. Regardless, I have tried to implement a QnA system using RAG for this project. 

## Overview of the project workflow:
1. Load the dataset from the CSV file
2. Perform filtering on the dataset (based on app version, word count, like count, and language)
3. Load the documents from the dataframe (text splitting is unnecessary due to the short length of each review)
4. Create Pinecone index to store the document embeddings along with their metadata
5. Create QnA pipeline (check below for more details)

## The QnA pipeline consists of the following steps:
1. Check if the question is flagged as inappropriate by the OpenAI Moderation API
2. Use MultiQueryRetriever to generate three different versions of the question and retrieve the relevant documents
3. Choose the top-n retrieved documents based on the number of likes
4. Inject the original question and retrieved documents into the prompt and pass the prompt to the LLM

## Limitations for this project:
- More suitable for open-ended questions like the one mentioned in the PDF
- Only include reviews for version 8.8.xx.xxx
- Emphasize the reviews with the highest like count (not necessarily a bad thing)

## Future improvement:
- In addition to moderation, also check the relevancy of the user question, so irrelevant questions will be filtered out early (before the document retrieval step)
- Adding memory to the pipeline so it can take into account the previous questions and answers
