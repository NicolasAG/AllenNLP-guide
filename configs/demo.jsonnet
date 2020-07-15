{
    "dataset_reader" : {
        "type": "classification-tsv",
        "tokenizer": {
            // "type": "whitespace"
            "type": "spacy",
            "pos_tags": true,
        },
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "namespace": "token_vocab", // vocabulary namespace
            },
            "token_characters": {
                "type": "characters",
                "namespace": "character_vocab", // vocabulary namespace
                "min_padding_length": 3, //maximum value of ngram_filter_sizes
            },
            "pos_tags": {
                "type": "single_id",
                "namespace": "pos_tag_vocab", // vocabulary namespace
                "feature_name": "tag_",
            }
        }
    },
    "train_data_path": "/allennlp/data/train.tsv",
    "validation_data_path": "/allennlp/data/dev.tsv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10,
                    "vocab_namespace": "token_vocab"
                },
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        // "type": "embedding",
                        "embedding_dim": 3,
                        "vocab_namespace": "character_vocab"
                    },
                    "encoder": {
                        "type": "cnn",
                        "embedding_dim": 3,
                        "num_filters": 4,
                        "ngram_filter_sizes": [3,]
                    }
                },
                "pos_tags": {
                    "type": "embedding",
                    "embedding_dim": 10,
                    "vocab_namespace": "pos_tag_vocab"
                }
            }
        },
        "encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": 24,
            "averaged": true
        }
    },
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 5,
        "cuda_device": 0,
    }
}