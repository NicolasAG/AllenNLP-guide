local bert_model = "bert-base-uncased";

{
    "dataset_reader" : {
        "type": "classification-tsv",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
                "namespace": "tags", // vocabulary namespace
            }
        },
        "max_tokens": 512
    },
    "train_data_path": "/allennlp/data/train.tsv",
    "validation_data_path": "/allennlp/data/dev.tsv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model,
                    "train_parameters": true,
                }
            }
        },
        "encoder": {
            "type": "bert_pooler",
            "pretrained_model": bert_model,
            "requires_grad": true,
        }
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 8,
            "sorting_keys": ['text']
        },
        // "shuffle": true
    },
    "validation_data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 8,
            "sorting_keys": ['text']
        },
        // "shuffle": false
    },
    "trainer": {
        "type": "gradient_descent",
        "serialization_dir": "/allennlp/models/tmp",
        "num_epochs": 5,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5,
        },
        "cuda_device": 0,
    }
}