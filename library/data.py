import torch
import numpy as np
from overrides import overrides

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers import DatasetReader, Seq2SeqDatasetReader
from allennlp.data.tokenizers import CharacterTokenizer, Tokenizer, Token
from allennlp.data.vocabulary import Vocabulary


import logging
logger = logging.getLogger(__name__)



@DatasetReader.register("addition_seq2seq_datasetreader")
class AdditionSeq2SeqDatasetReader(Seq2SeqDatasetReader):
    def __init__(self):
        super().__init__(lazy=False)
        self._source_tokenizer = CharacterTokenizer()
        self._target_tokenizer = CharacterTokenizer()

    @overrides
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
