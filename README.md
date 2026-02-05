# App Engagement Analysis

## Purpose
The project focuses on processing app engagement data from January and February (2024). It includes graph construction, model training (or pre-trained weights), and prediction workflows. The primary tool is a Jupyter notebook for loading data, models, and generating predictions.

## Repository Structure
- **`Jan-Feb-parquet/`**: Directory containing Parquet files with raw or processed data for January and February. These are user interaction logs or engagement metrics in columnar format for efficient querying.
- **`Prediction Probabilities.docx`**: A Microsoft Word document detailing prediction probabilities from the model runs, for reviewing output results.
- **`jan_graph.pt`**: PyTorch serialized graph object for January data, probably a heterogeneous graph representing users, apps, and interactions.
- **`jan_mappings.pt`**: PyTorch tensor or dictionary mappings (e.g., node IDs to entities) for the January graph.
- **`metapath2vec_january.pth`**: Pre-trained model weights for a MetaPath2Vec embedding model trained on January data. MetaPath2Vec is used for learning representations in heterogeneous graphs.
- **`model_prediction.ipynb`**: Jupyter notebook implementing the prediction pipeline. It loads the graph, mappings, and model to compute engagement predictions.

## Model Details
- **MetaPath2Vec**: A graph embedding technique for heterogeneous networks. Here, it's applied to app engagement data, modeling paths like user-app-session. Using meta-path-guided random walks to sample heterogeneous neighborhoods and a skip-gram model to learn embeddings that preserve structural and semantic relationships.
- Predictions: The notebook outputs engagement probabilities.

## Reference
Yuxiao Dong, Nitesh V. Chawla, and Ananthram Swami. 2017. metapath2vec: Scalable Representation Learning for Heterogeneous Networks. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '17), August 13--17, 2017, Halifax, NS, Canada. ACM, New York, NY, USA, 135--144. DOI:https://doi.org/10.1145/3097983.3098036
