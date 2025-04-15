
from Nexus.training.reranker.recommendation.runner import RankerRunner
from Nexus.training.reranker.recommendation.modeling import MLPRanker, DCNv2Ranker


def main():
    data_config_path = "./examples/recommendation/config/data/recflow_ranker_local_seq.json"
    train_config_path = "./examples/recommendation/config/DCN/train.json"
    model_config_path = "./examples/recommendation/config/DCN/model.json"
    
    runner = RankerRunner(
        model_config_or_path=model_config_path,
        data_config_or_path=data_config_path,
        train_config_or_path=train_config_path,
        model_class=DCNv2Ranker
    )
    runner.run()


if __name__ == "__main__":
    main()
