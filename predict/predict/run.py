import json
import argparse
import os
import time
from collections import OrderedDict

from keras.models import load_model
from numpy import argsort

from preprocessing.preprocessing.embeddings import embed

import logging

logger = logging.getLogger(__name__)


class TextPredictionModel:
    def __init__(self, model, params, labels_to_index):
        self.model = model
        self.params = params
        self.index_to_label = labels_to_index
        self.labels_index_inv = {ind: lab for lab, ind in self.index_to_label.items()}

    @classmethod
    def from_artefacts(cls, artefacts_path: str):
        """
            from training artefacts, returns a TextPredictionModel object
            :param artefacts_path: path to training artefacts
        """
        # load model
        model = load_model(os.path.join(artefacts_path, 'model.h5'))

        # load params
        with open(os.path.join(artefacts_path, 'params.json'), 'r') as f:
            params = json.load(f)

        # load labels_to_index
        with open(os.path.join(artefacts_path, 'labels_index.json'), 'r') as f:
            labels_to_index = json.load(f)

        return cls(model, params, labels_to_index)

    def predict(self, text_list, top_k=5):
        """
            predict top_k tags for a list of texts
            :param text_list: list of text (questions from stackoverflow)
            :param top_k: number of top tags to predict
        """
        tic = time.time()

        logger.info(f"Predicting text_list=`{text_list}`")

        # embed text_list
        embeddings = embed(text_list)

        # predict tags indexes from embeddings
        predictions = self.model.predict(embeddings)

        # from tags indexes compute top_k tags for each text
        predictions = [argsort(pred)[-top_k:][::-1] for pred in predictions]

        # Convert indexes to labels
        predictions_with_labels = [[self.index_to_label[str(index)] for index in pred] for pred in predictions]

        logger.info("Prediction done in {:2f}s".format(time.time() - tic))

        return predictions_with_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artefacts_path", help="path to trained model artefacts")
    parser.add_argument("text", type=str, default=None, help="text to predict")
    args = parser.parse_args()

    logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)

    model = TextPredictionModel.from_artefacts(args.artefacts_path)

    if args.text is None:
        while True:
            txt = input("Type the text you would like to tag: ")
            predictions = model.predict([txt])
            print(predictions)
    else:
        print(f'Predictions for `{args.text}`')
        print(model.predict([args.text]))
