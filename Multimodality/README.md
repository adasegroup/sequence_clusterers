## Multimodality
To execute code run multimodality.sh, adding necessary parameters if needed. 

	Tex_embed.py - creates cnn embeddings for text part of dataset (refer to https://arxiv.org/pdf/1408.5882.pdf). 

	Seq_embed - creates embeding, using RMTPP (refer to https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf). 

	Cluster.py - concatenates embeddings of text and seq and clusterizes it with Kmeans.
