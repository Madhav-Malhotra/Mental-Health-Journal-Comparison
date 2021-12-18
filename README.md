# Mental Health Journal Comparison
This Flask API compares mental health reflections for similarities.

I was inspired to build it after thinking about how I found it helpful to talk about mental health issues with others who'd had similar experiences. That made me wonder: "How could we connect people with similar mental health issues together to talk?"

So, I thought about comparing people's journal entries for similarities and then using those to connect people together. I used the [Pytorch Infersent](https://github.com/facebookresearch/InferSent) model to generate sentence-level word embeddings from journal entries. Then, I used cosine similarity to compare different word embeddings. And I implemented this algorithm on Flask API that could be called by some front-end web app.

In the repository, the `/data/` folder contains parameters for the Infersent model. `model.py` has the impementation of the Infersent model on Pytorch. `api.py` has my implementation of the Cosine Similarity algorithm and the Flask API. 
