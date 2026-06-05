# Small Story Generator LLM
A small and lightweight decoder only Large Language Model with approximately 76.9 million parameters buit for small and creative story generation as part of a university course assignment.

# Contents of this Project
This repository contains the code for the project which consists of two files:

1. **'train.py'** 
2. **'fine_tuning.py'** 

Both do as their names suggest. The first one is for training the model and the second for fine tuning it on a different dataset, to apply a particular style to it.

# Model Details
The model has been coded in a pytorch enviroment and it has a decoder only architecture. It is based on the **Attention is all you need** paper while also being heavily inspired from the video **Let's build GPT: from scratch, in code, spelled out** from the youtube channel **Andrej Karpathy.** The model also has approximately only 76.9 million parameters due to the intense-computing nature of llms making it very fast and efficient but also not that good of a story generator model.

# Credits
**Video: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1s**
**Paper: https://arxiv.org/abs/1706.03762**

# Contributions
Even though I am largely finished with this project feel free to suggest improvements, address issues or fork the repository to experiment with your own fine-tuning and datasets.
