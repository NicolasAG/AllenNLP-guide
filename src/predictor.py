from allennlp.common import JsonDict
from allennlp.predictors import Predictor
from allennlp.data import Instance

# copied from https://guide.allennlp.org/training-and-prediction#4

"""
For making predictions in a demo setting, AllenNLP uses Predictors, which are a thin wrapper around your trained model.
A Predictor’s main job is to take a JSON representation of an instance,
convert it to an Instance using the dataset reader (see text_to_instance() in custom data reader),
pass it through the model, and return the prediction in a JSON serializable format.

AllenNLP provides implementations of Predictors for common tasks.
In fact, it includes TextClassifierPredictor, a generic Predictor for text classification tasks,
so you don’t even need to write your own!
Here, we are writing one from scratch solely for demonstration,
but you should always check whether the predictor for your task is already there.
"""


@Predictor.register("sentence_classifier")
class SentenceClassifierPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)  # see ClassificationTsvReader class in data.py
