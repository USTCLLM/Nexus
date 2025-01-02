from UniRetrieval.training.embedder.recommendation.runner import RetrieverRunner
from UniRetrieval.training.embedder.recommendation.modeling import MLPRetriever


def main():
    data_config_path = "/data1/home/recstudio/haoran/UniRetrieval/recommender_examples/config/data/recflow_retriever.json"
    train_config_path = "/data1/home/recstudio/haoran/UniRetrieval/recommender_examples/config/mlp_retriever/train.json"
    model_config_path = "/data1/home/recstudio/haoran/UniRetrieval/recommender_examples/config/mlp_retriever/model.json"
    
    runner = RetrieverRunner(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        train_config_path=train_config_path,
        model_class=MLPRetriever,
    )
    runner.run()


if __name__ == "__main__":
    main()
