local embedding_dim = 15;
local hidden_dim = 128;
local num_epochs = 10;
local patience = 10;
local batch_size = 2;
local learning_rate = 0.1;

{
  "dataset_reader": {
    "type": "addition_seq2seq_datasetreader"
  },
  "train_data_path": "data/train_100000_1000.csv",
  "validation_data_path": "data/val_300.csv",
  "model": {
    "type": "simple_seq2seq",
    "max_decoding_steps": 5,
    "source_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": embedding_dim
        }},
      "encoder": {
        "type": "lstm",
        "input_size": embedding_dim,
        "hidden_size": hidden_dim
      }
    },
    "iterator": {
      "type": "bucket",
      "batch_size": batch_size,
      "sorting_keys": [["source_tokens", "num_tokens"]]
    },
  "trainer": {
    "cuda_device": -1,
    "num_epochs": num_epochs,
    "optimizer": {
      "type": "adam",
    },
    "patience": patience,
    "validation_metric": "+accuracy"
  }
}
