python preprocess_data.py \
       --input gpt2_data/webtext.train.jsonl \
       --output-prefix gpt2_data/my-gpt2 \
       --vocab gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod \
       --split-sentences \
       --workers 4 \
       --chunk-size 128

