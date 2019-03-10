## Addition with Encoder-Decoder using AllenNLP

The same idea as in [https://github.com/mfa/addition_seq2seq/](https://github.com/mfa/addition_seq2seq/)
but using [allennlp](https://github.com/allenai/allennlp/) library.

This code requires Python 3.6.

### Install

pip install -r code/requirements.txt

### Generate Data

see [https://github.com/mfa/addition_seq2seq/](https://github.com/mfa/addition_seq2seq/)


### tensorboard

```
docker run -p 0.0.0.0:6006:6006 -it -v `pwd`/output/log:/root/logs  tensorflow/tensorflow tensorboard --logdir /root/logs/
```
