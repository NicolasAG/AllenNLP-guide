import logging
import torch

from allennlp.data import DataLoader, Vocabulary
from allennlp.data.tokenizers import (
    CharacterTokenizer,
    PretrainedTransformerTokenizer,
    SpacyTokenizer,
    WhitespaceTokenizer,
)
from allennlp.data.token_indexers import (
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
    ELMoTokenCharactersIndexer,
    PretrainedTransformerIndexer,
    PretrainedTransformerMismatchedIndexer,
)
from allennlp.data.samplers import BucketBatchSampler
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import (
    Embedding,
    TokenCharactersEncoder,
    ElmoTokenEmbedder,
    PretrainedTransformerEmbedder,
    PretrainedTransformerMismatchedEmbedder,
)
from allennlp.modules.seq2vec_encoders import (
    BagOfEmbeddingsEncoder,
    BertPooler,
    CnnEncoder
)
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer, HuggingfaceAdamWOptimizer
from allennlp.training.util import evaluate

from data import ClassificationTsvReader
from model import SimpleClassifier
from predictor import SentenceClassifierPredictor

# updated from https://guide.allennlp.org/training-and-prediction#1


logging.basicConfig(format='%(asctime)s [%(process)d] %(name)s.%(module)s.%(funcName)s l%(lineno)d %(levelname)s | %(message)s', level=logging.INFO)


def build_data_reader(bert_model: str = None):
    if bert_model:
        tokenizer = PretrainedTransformerTokenizer(model_name=bert_model)
        token_indexers = {"bert": PretrainedTransformerIndexer(model_name=bert_model, namespace='tags')}
        # use confusing default value of 'tags' so that we don't add padding or UNK tokens to this namespace,
        # which would break on loading because we wouldn't find our default OOV token.
        max_tokens = 512
    else:
        # (1) how to split strings into tokens: WhiteSpace or Spacy
        #tokenizer = WhitespaceTokenizer()
        tokenizer = SpacyTokenizer(pos_tags=True)  # also set the '_tag' variable of each Token
        # (2) how to convert Token strings into Token IDs: 3 ways so 3 distinct vocab namespaces
        # will add tokens to vocab namespaces 'token_vocab', 'character_vocab', and 'pos_tag_vocab'
        token_indexers = {
            'tokens': SingleIdTokenIndexer(namespace='token_vocab'),
            'token_characters': TokenCharactersIndexer(namespace='character_vocab', min_padding_length=3),
            #                                                                       ^
            # We use this value as the minimum length of padding. Usually used with CnnEncoder,
            # its value should be set to the maximum value of ngram_filter_sizes correspondingly.
            'pos_tags': SingleIdTokenIndexer(namespace='pos_tag_vocab', feature_name='tag_')
            #                                                           ^
            # We will use the Token._tag as input instead of the default token.text attribute
        }
        max_tokens = None  # 64
    '''
    dataset_reader = ClassificationTsvReader(tokenizer, token_indexers, max_tokens, cache_directory='data_cache')
    DatasetReaders can cache datasets by serializing created instances and writing them to disk.
    The next time the same file is requested the instances are deserialized from the disk instead of
    being created from the file.

    Instances are serialized by jsonpickle by default, although you can override this behavior if you want.
    To do this, either override the serialize_instance and deserialize_instance
    methods in your DatasetReader (if you want a one-instance-per-line serialization),
    or the _instances_to_cache_file and _instances_from_cache_file
    methods (if you want something that is more efficient to store and read).

    The objects that get stored can be pretty large, so this is often only useful if you have
    particularly slow preprocessing.

    ?? BUG ?? reloading cache_directory instances fails when using bert models :/
    '''
    return ClassificationTsvReader(tokenizer=tokenizer, token_indexers=token_indexers, max_tokens=max_tokens)


def build_model(vocab: Vocabulary, bert_model: str = None) -> Model:
    if bert_model:
        embedder = BasicTextFieldEmbedder({"bert": PretrainedTransformerEmbedder(model_name=bert_model,
                                                                                 train_parameters=True)})
        encoder = BertPooler(pretrained_model=bert_model, requires_grad=True)
    else:
        # (3) How to get vectors for each Token ID:
        # (3.1) embed each token
        token_embedding = Embedding(embedding_dim=10, num_embeddings=vocab.get_vocab_size("token_vocab"))
        # pretrained_file='https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.50d.txt.gz'

        # (3.2) embed each character in each token
        character_embedding = Embedding(embedding_dim=3, num_embeddings=vocab.get_vocab_size("character_vocab"))
        cnn_encoder = CnnEncoder(embedding_dim=3, num_filters=4, ngram_filter_sizes=[3,])
        token_encoder = TokenCharactersEncoder(character_embedding, cnn_encoder)
        # (3.3) embed the POS of each token
        pos_tag_embedding = Embedding(embedding_dim=10, num_embeddings=vocab.get_vocab_size("pos_tag_vocab"))

        # Each TokenEmbedders embeds its input, and the result is concatenated in an arbitrary (but consistent) order
        # cf: https://docs.allennlp.org/master/api/modules/text_field_embedders/basic_text_field_embedder/
        embedder = BasicTextFieldEmbedder(
            token_embedders={"tokens": token_embedding,
                             "token_characters": token_encoder,
                             "pos_tags": pos_tag_embedding}
        )  # emb_dim = 10 + 4 + 10 = 24
        encoder = BagOfEmbeddingsEncoder(embedding_dim=24, averaged=True)
        #                                                  ^
        # average the embeddings across time, rather than simply summing
        # (ie. we will divide the summed embeddings by the length of the sentence).
    return SimpleClassifier(vocab, embedder, encoder)


'''
from: https://guide.allennlp.org/representing-text-as-features#4

Putting together the data and model side of TextFields in AllenNLP
requires coordinating the keys used in a few different places:
  (1) the vocabulary namespaces used by the TokenIndexers
      and the TokenEmbedders need to match (where applicable),
      so that you get the right number of embeddings for each
      kind of input, and
  (2) the keys used for the TokenIndexer dictionary in the TextField
      need to match the keys used for the TokenEmbedder dictionary
      in the BasicTextFieldEmbedder.
- - -

from https://guide.allennlp.org/representing-text-as-features#5

Typically, there is a one-to-one relationship between a TokenIndexer and a TokenEmbedder,
and each TokenIndexer mostly only makes sense with one Tokenizer.
Below is a mostly-exhaustive list of the options available in AllenNLP:

Using a word-level tokenizer (such as SpacyTokenizer or WhitespaceTokenizer):
- SingleIdTokenIndexer -> Embedding (for things like GloVe or other simple embeddings, including learned POS tag embeddings)
- TokenCharactersIndexer -> TokenCharactersEncoder (for things like a character CNN)
- ElmoTokenIndexer -> ElmoTokenEmbedder (for ELMo)
- PretrainedTransformerMismatchedIndexer -> PretrainedTransformerMismatchedEmbedder
(for using a transformer like BERT when you really want to do modeling at the word level)

Using a character-level tokenizer (such as CharacterTokenizer):
- SingleIdTokenIndexer -> Embedding

Using a wordpiece tokenizer (such as PretrainedTransformerTokenizer):
- PretrainedTransformerIndexer -> PretrainedTransformerEmbedder
- SingleIdTokenIndexer -> Embedding (if you don’t want contextualized wordpieces for some reason)
'''


def build_trainer(model: Model, ser_dir: str, train_loader: DataLoader, valid_loader: DataLoader,
                  hugging_optim: bool, cuda_device: int) -> Trainer:
    params = [ [n, p] for n, p in model.named_parameters() if p.requires_grad ]
    logging.info(f"{len(params)} parameters requiring grad updates")
    if hugging_optim:
        optim = HuggingfaceAdamWOptimizer(params, lr=1.0e-5)
    else:
        optim = AdamOptimizer(params)
    return GradientDescentTrainer(
        model=model,
        serialization_dir=ser_dir,
        data_loader=train_loader,
        validation_data_loader=valid_loader,
        num_epochs=5,
        patience=None,  # early stopping is disabled
        optimizer=optim,
        cuda_device=cuda_device
    )


def run_training_loop(bert_model=None):
    # BUILDING DATA READER
    logging.info("Building data reader...")
    dataset_reader = build_data_reader(bert_model)

    logging.info("Reading data...")
    # These are a subclass of pytorch Datasets, with some allennlp-specific functionality added.
    train_instances = dataset_reader.read("/allennlp/data/train.tsv")
    logging.info(f"got {len(train_instances)} train instances")
    valid_instances = dataset_reader.read("/allennlp/data/dev.tsv")
    logging.info(f"got {len(valid_instances)} valid instances")

    logging.info("Building vocabulary...")
    vocab = Vocabulary.from_instances(train_instances + valid_instances,
                                      min_count={'text': 1})
    # all tokens that appear at least once in namespace 'tokens'
    logging.info(vocab)
    # for namespace in vocab.get_namespaces():
    #     logging.info(f"vocab[{namespace}] size: {vocab.get_vocab_size(namespace=namespace)}")
    # return

    logging.info("Building model...")
    model = build_model(vocab, bert_model)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    logging.info(model)

    logging.info("Building data loaders...")
    # This is the allennlp-specific functionality in the Dataset object:
    # we need to be able convert strings in the data to integers.
    # this is how we do it.
    train_instances.index_with(vocab)
    valid_instances.index_with(vocab)

    # Using a BucketBatchSampler:
    # It sorts the instances by the length of their longest Field (or by any sorting keys you specify)
    # and automatically groups them so that instances of similar lengths get batched together.
    train_batch_sampler = BucketBatchSampler(train_instances, batch_size=8,
                                             sorting_keys=['text'])  # sort by length of instance field 'text'
    valid_batch_sampler = BucketBatchSampler(valid_instances, batch_size=8,
                                             sorting_keys=['text'])  # sort by length of instance field 'text'

    # These are again a subclass of pytorch DataLoaders, with an allennlp-specific collate function,
    # that runs our indexing and batching code.
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is what actually does indexing and batching.
    #train_loader = DataLoader(train_instances, batch_size=8, shuffle=True)
    #valid_loader = DataLoader(valid_instances, batch_size=8, shuffle=False)
    train_loader = DataLoader(train_instances, batch_sampler=train_batch_sampler) #, shuffle=True)
    valid_loader = DataLoader(valid_instances, batch_sampler=valid_batch_sampler) #, shuffle=False)

    logging.info("Building trainer...")
    trainer = build_trainer(model, "/allennlp/models/tmp", train_loader, valid_loader,
                            hugging_optim=bert_model is not None, cuda_device=cuda_device)
    logging.info("Start training...")
    trainer.train()
    logging.info("done.")

    '''
    logging.info("====================")
    logging.info("Loading test data...")
    test_instances = dataset_reader.read("/allennlp/data/test.tsv")
    test_instances.index_with(vocab)
    test_loader = DataLoader(test_instances, batch_size=8, shuffle=False)

    logging.info("Predicting on test set...")
    # utility function to run your model and get the metric on the test set
    results = evaluate(model, test_loader)
    logging.info(results)
    '''

    # from: https://guide.allennlp.org/training-and-prediction#4
    logging.info("====================")
    logging.info("Constructing a Predictor for custom inputs...")
    predictor = SentenceClassifierPredictor(model, dataset_reader)

    for sent in ['A good movie!', 'This was a monstrous waste of time.']:
        logging.info("")
        logging.info(f"Predicting for '{sent}'...")
        output = predictor.predict(sent)
        for label_id, prob in enumerate(output['probs']):  # ['probs'] coming from model's forward() function
            # Because the returned result (output['probs']) is just an array of probabilities for class labels,
            # we use vocab.get_token_from_index() to convert a label ID back to its label string.
            logging.info(f"{vocab.get_token_from_index(label_id, 'labels')}: {prob}")

    logging.info("done.")


if __name__ == '__main__':
    run_training_loop()

    # Any transformer model name that huggingface's transformers library supports will work here.
    # Under the hood, we're grabbing pieces from huggingface.
    # https://huggingface.co/models
    #run_training_loop(bert_model="bert-base-uncased")
