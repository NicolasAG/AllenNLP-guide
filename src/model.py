from typing import Dict

import logging
import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy

from data import ClassificationTsvReader

# copied from https://guide.allennlp.org/your-first-model#6


@Model.register('simple_classifier')
class SimpleClassifier(Model):
    def __init__(self, vocab: Vocabulary,
                       embedder: TextFieldEmbedder,
                       encoder: Seq2VecEncoder,
                       initializer: InitializerApplicator = None,
                       regularizer: RegularizerApplicator = None):
        super().__init__(vocab, regularizer=regularizer)
        logging.info(f"vocab passed to model: {vocab}")

        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(),
                                          num_labels)
        self.accuracy = CategoricalAccuracy()
        '''
        each Metric instance holds "counts" that are necessary and sufficient to compute the metric.
        For accuracy, these counts are the number of total predictions as well as the number of correct predictions.
        These counts get updated after every call to the instance itself, i.e., the self.accuracy(logits, label) line.
        To pull out the computed metric, call get_metrics() with a flag specifying whether to reset the counts.
        '''

        # from https://guide.allennlp.org/building-your-model#4
        if initializer:
            initializer(self)

    # copied from https://guide.allennlp.org/your-first-model#7
    def forward(self, text: Dict[str, torch.Tensor],
                      label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        when making predictions for unseen inputs, the instances are unlabeled.
        By making the label parameter optional, the model can support both cases.
        :param text: instance text field with the SAME name
        :param label: instance label field with the SAME name

        from https://guide.allennlp.org/representing-text-as-features#9
        :param text: If you use a TextField in your data processing code with AllenNLP,
        you will get a TextFieldTensors object back, which is a complex dictionary structure.
        The nice thing about this structure is that it is flexible, allowing you to change
        the underlying representation without changing your model code.
        But that flexibility can also be a hinderance to usability sometimes.
        e strongly recommend against writing code that directly accesses the internals of a TextFieldTensors object.
        When you do this, you hard-code assumptions about what representation you used,
        and you make it very difficult to change those representations later.
        """
        # The TextFieldEmbedder object converts the TextFieldTensors objet into one embedded vector per input token.
        embedded_text = self.embedder(text)    # ~(bs, max_len, emb_dim)

        # The other common operations that you might want to do with a TextFieldTensors
        # object are to get a mask out of it (for use in modeling operations),
        # and to get the token ids (either for use in some language-modeling-like task,
        # or for displaying them to a human after converting them back to strings).
        # allennlp.nn.util provides utility functions for both of these operations:
        # - get_text_field_mask
        # - get_token_ids_from_text_field_tensors
        mask = util.get_text_field_mask(text)  # ~(bs, max_len) of booleans
        #token_ids = util. get_token_ids_from_text_field_tensors(text)  # ~(bs, max_len) of token IDs

        # Padding gives us fixed length tensors for a given batch,
        # but it means that we have portions of our input that are actually empty values.
        # We need to know which values are actually empty, so that we don’t assign probability mass to those tokens.
        # AllenNLP handles *some of it* inside of a TextFieldEmbedder.
        encoded_text = self.encoder(embedded_text, mask)  # ~(bs, enc_dim)
        logits = self.classifier(encoded_text)  # ~(bs, n_class)
        probs = torch.nn.functional.softmax(logits, dim=-1)  # ~(bs, n_class)
        output = {'probs': probs}
        if label is not None:
            output['loss'] = torch.nn.functional.cross_entropy(logits, label)  # ~(1,)
            self.accuracy(logits, label)  # update the metric by feeding the prediction and the gold labels
        return output

    # updated from https://guide.allennlp.org/training-and-prediction#3
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


if __name__ == '__main__':

    def build_model(vocab: Vocabulary) -> Model:
        vocab_size = vocab.get_vocab_size("tokens")  # "tokens" from data_reader.token_indexers ??
        embedder = BasicTextFieldEmbedder({"tokens": Embedding(embedding_dim=10,
                                                               num_embeddings=vocab_size)})
        encoder = BagOfEmbeddingsEncoder(embedding_dim=10)
        return SimpleClassifier(vocab, embedder, encoder)

    dataset_reader = ClassificationTsvReader()  # max_tokens=64

    print("Reading data...")
    instances = dataset_reader.read("/allennlp/data/train.tsv")

    print("Building vocabulary...")
    vocab = Vocabulary.from_instances(instances)
    print("Building model...")
    model = build_model(vocab)

    print("Running the model...")
    outputs = model.forward_on_instances(instances[:4])
    print(outputs)
