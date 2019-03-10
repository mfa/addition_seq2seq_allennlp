#!/bin/bash

OUTPUT=output/`date --iso-8601=seconds`
echo $OUTPUT

allennlp train addition.jsonnet -s $OUTPUT --include-package library
