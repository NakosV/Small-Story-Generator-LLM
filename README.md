# Small Story Generator LLM
A small and lightweight decoder-only Large Language Model with approximately 76.9 million parameters buit for small and creative story generation as part of a university course assignment.

# Contents of this Project
This repository contains the code for the project which consists of the two following files:

1. **[`LLM-BPE.py`](LLM-BPE.py)** 
2. **[`LLM-Generate.py`](LLM-Generate.py)** 

Both do as their names suggest. The first one is for training the model from scratch on a dataset and the second is for fine tuning it on a different dataset, to apply a particular writing style to it.

# Model Details
The model has been built in a Pytorch environment and it has a decoder-only architecture. It is based on the **Attention is all you need** paper while also being heavily inspired from the video **Let's build GPT: from scratch, in code, spelled out** from the Youtube channel **Andrej Karpathy.** This model is small, having approximately **76.9 million** parameters. Because of the massive computing power that is needed to train an LLM model from scratch I had to keep the model small, making it very fast and efficient when used, while also being able to be trained at Google Colab on the free plan without exceeding the time limit of a session, with the cost being that the model isn't that good of a story generator. But of course this model can easily be scaled up or down just by changing some variables.

# Results
Here I have some information on what the model was trained on and some results
**[You can click here to check them out](RESULTS.md)**

# Credits
- **Video: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1s**
- **Paper: https://arxiv.org/abs/1706.03762**

# Contributions
Even though I am largely finished with this project feel free to suggest improvements, address issues or fork the repository to experiment with your own fine-tuning and datasets.
