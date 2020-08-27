# KGVAE
project GCN-VAE for knowledge graphs

# Abstract
Knowledge graphs are powerful abstraction to represent relational facts among entities. Since most real world knowledge graphs are manually collected and largely incomplete, predicting missing links from given graph becomes crucial for knowledge graph to be more useful. Recently many studies have shown promising results by embedding entities and relation in a vector space. However, a common issue with most point estimates methods is that they could fail to capture more complex structures such as one-to-many, or many-to-many relations, nor can they adapt to scenarios where entities and relations have intrinsic uncertainties and multiple semantic meanings. To this end, we consider a probabilistic framework and propose GCN-VAE, a variantaional autoencoder to encode entities' neighborhood structure into a compact latent embedding distribution and reconstruct the links between entities by estimating edge likelihood via a decoder. We demonstrated the effectiveness of GCN-VAE on two popular benchmarks and got improved performance over state-of-the-art baselines on FB15k-237. 

