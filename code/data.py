import torch
import numpy as np

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers import DatasetReader, Seq2SeqDatasetReader
from allennlp.data.tokenizers import CharacterTokenizer, Tokenizer, Token
from allennlp.data.vocabulary import Vocabulary


import logging

logger = logging.getLogger(__name__)

torch.manual_seed(1)


@DatasetReader.register("addition-seq2seq")
class AdditionSeq2SeqDatasetReader(Seq2SeqDatasetReader):
    def __init__(self):
        super().__init__(lazy=False)
        self._source_tokenizer = CharacterTokenizer()
        self._target_tokenizer = CharacterTokenizer()

    def _read(self, file_path):
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n").replace('"', "")

                if not line:
                    continue

                line_parts = line.split(",")
                if len(line_parts) != 2:
                    raise ConfigurationError(
                        "Invalid line format: %s (line number %d)"
                        % (line, line_num + 1)
                    )
                source_sequence, target_sequence = line_parts
                yield self.text_to_instance(source_sequence, target_sequence)


reader = AdditionSeq2SeqDatasetReader()
train_dataset = reader.read("../data/train_100000_1000.csv")
validation_dataset = reader.read("../data/val_300.csv")

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

print(vocab.print_statistics())
print(train_dataset[0])
