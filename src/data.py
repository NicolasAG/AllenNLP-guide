from typing import Dict, Iterable

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer


# copied from https://guide.allennlp.org/your-first-model#4
# updated from https://guide.allennlp.org/training-and-prediction#1

@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    """
    Dataset Reader for a tab separated file of the form:
    [text] TAB [label]
    """
    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 max_tokens: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_tokens = max_tokens

    # updated from https://guide.allennlp.org/training-and-prediction#4
    '''
    refactor the logic for creating an Instance from tokens and label
    making this piece of code sharable between DatasetReader and Predictor.
    Building two pipelines: one for training and another for prediction.
    By factoring out the common logic for creating instances and sharing it between two pipelines,
    we are making the system less susceptible to any issues arising from possible discrepancies
    in how instances are created between the two, a problem known as training-serving skew.
    '''
    def text_to_instance(self, text: str, label: str = None) -> Instance:
        """
        when making predictions for unseen inputs, the instances are unlabeled.
        By making the label parameter optional, the dataset reader can support both cases.
        """
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields = {'text': text_field}
        if label:
            fields['label'] = LabelField(label, label_namespace='labels')  # vocab namespace
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                text, sentiment = line.strip().split('\t')
                yield self.text_to_instance(text, sentiment)


if __name__ == '__main__':
    dataset_reader = ClassificationTsvReader(WhitespaceTokenizer(), {'tokens': SingleIdTokenIndexer()}, max_tokens=64)
    instances = dataset_reader.read("/allennlp/data/train.tsv")
    for instance in instances[:10]:
        print(instance)
