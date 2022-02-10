Multimodality
To execute code run multimodality.sh, adding necessary parameters if needed. 
Tex_embed.py - create embeddings for text part of dataset (like here https://arxiv.org/pdf/1408.5882.pdf), using cnn. Seq_embed - creates embeding, using RMTPP (like here https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf). 
Cluster.py - concatenate embeddings of text and seq and clusterize it witth Kmeans.
