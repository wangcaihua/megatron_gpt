# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""
import os
import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, BertModel, T5Model, ModelType
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter as DataLoaderIter


tokenizer_dict = {
    'gpt': 'GPT2BPETokenizer',
    'bert': 'BertWordPieceLowerCase',
    't5': 'BertWordPieceLowerCase'
}

keys_dict = {
    'gpt': ['text'],
    'bert': ['text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask'],
    't5': ['text_enc', 'text_dec', 'labels', 'loss_mask', 'enc_mask', 'dec_mask', 'enc_dec_mask']
}


def init_mpi_env():
    # for key in os.environ:
    #     if key.startswith('OMPI_'):
    #         print(key, os.environ[key])
    #     elif key in {'MASTER_ADDR', 'MASTER_PORT'}:
    #         print(key, os.environ[key])

    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = os.environ.get('OMPI_COMM_WORLD_SIZE', '1')
    if 'RANK' not in os.environ:
        os.environ['RANK'] = os.environ.get('OMPI_COMM_WORLD_RANK', '0')
    if 'LOCAL_SIZE' not in os.environ:
        os.environ['LOCAL_SIZE'] = os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', '1')
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
    if 'NODE_RANK' not in os.environ:
        os.environ['NODE_RANK'] = os.environ.get('OMPI_COMM_WORLD_NODE_RANK', '0')

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '9876')


def model_provider(pre_process=True, post_process=True,
                   add_encoder=True, add_decoder=True):
    """Build the model."""

    args = get_args()
    print_rank_0(f'building {args.nlp_model_type} model ...')
    if args.nlp_model_type == 'gpt':
        model = GPTModel(
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )
    elif args.nlp_model_type == 'bert':
        num_tokentypes = 2 if args.bert_binary_head else 0
        model = BertModel(
            num_tokentypes=num_tokentypes,
            add_binary_head=args.bert_binary_head,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process)
    elif args.nlp_model_type == 't5':
        model = T5Model(num_tokentypes=0,
                        parallel_output=True,
                        pre_process=pre_process,
                        post_process=post_process,
                        add_encoder=add_encoder,
                        add_decoder=add_decoder)
    else:
        raise Exception('nlp_model_type error!')
    return model


def get_batch(data_iterator: DataLoaderIter):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = keys_dict[args.nlp_model_type]
    datatype = torch.int64

    # Broadcast data.
    # print('data_iterator', data_iterator)
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    if args.nlp_model_type == 'gpt':
        # Unpack.
        tokens_ = data_b['text'].long()
        labels = tokens_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()
        # print('label', labels.size(),    # torch.Size([4, 1024])
        #       'tokens', tokens.size())   # torch.Size([4, 1024])

        # Get the masks and postition ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)
        # print('loss_mask', loss_mask.size(),              # torch.Size([4, 1024])
        #       'attention_mask', attention_mask.size(),    # torch.Size([1, 1, 1024, 1024])
        #       'position_ids', position_ids.size())        # torch.Size([4, 1024])
        return tokens, labels, loss_mask, attention_mask, position_ids
    elif args.nlp_model_type == 'bert':
        # Unpack.
        tokens = data_b['text'].long()
        types = data_b['types'].long()
        sentence_order = data_b['is_random'].long()
        loss_mask = data_b['loss_mask'].float()
        lm_labels = data_b['labels'].long()
        padding_mask = data_b['padding_mask'].long()

        return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask
    elif args.nlp_model_type == 't5':
        # Unpack.
        tokens_enc = data_b['text_enc'].long()
        tokens_dec = data_b['text_dec'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()

        enc_mask = (data_b['enc_mask'] < 0.5)
        dec_mask = (data_b['dec_mask'] < 0.5)
        enc_dec_mask = (data_b['enc_dec_mask'] < 0.5)

        return tokens_enc, tokens_dec, loss_mask, labels, \
            enc_mask, dec_mask, enc_dec_mask
    else:
        raise Exception('nlp_model_type error!')


def gpt_loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    # print('output_tensor', output_tensor, output_tensor.size())
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def bert_loss_func(loss_mask, sentence_order, output_tensor):
    lm_loss_, sop_logits = output_tensor

    lm_loss_ = lm_loss_.float()
    loss_mask = loss_mask.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    if sop_logits is not None:
        sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(),
                                   sentence_order.view(-1),
                                   ignore_index=-1)
        sop_loss = sop_loss.float()
        loss = lm_loss + sop_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss, sop_loss])
        return loss, {'lm loss': averaged_losses[0],
                      'sop loss': averaged_losses[1]}

    else:
        loss = lm_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss])
        return loss, {'lm loss': averaged_losses[0]}


def t5_loss_func(loss_mask, output_tensor):
    lm_loss_ = output_tensor.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    loss = lm_loss
    averaged_losses = average_losses_across_data_parallel_group([lm_loss])

    return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    batch_data = get_batch(data_iterator)
    timers('batch-generator').stop()

    if args.nlp_model_type == 'gpt':
        tokens, labels, loss_mask, attention_mask, position_ids = batch_data
        output_tensor = model(tokens, position_ids, attention_mask,
                            labels=labels)
        return output_tensor, partial(gpt_loss_func, loss_mask)
    elif args.nlp_model_type == 'bert':
        tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = batch_data
        if not args.bert_binary_head:
            types = None
        output_tensor = model(tokens, padding_mask, tokentype_ids=types,
                            lm_labels=lm_labels)
        return output_tensor, partial(bert_loss_func, loss_mask, sentence_order)
    elif args.nlp_model_type == 't5':
        tokens_enc, tokens_dec, loss_mask, lm_labels, enc_mask, dec_mask, enc_dec_mask = batch_data
        output_tensor = model(tokens_enc,
                              tokens_dec,
                              enc_mask,
                              dec_mask,
                              enc_dec_mask,
                              tokentype_ids=None,
                              lm_labels=lm_labels)
        return output_tensor, partial(t5_loss_func, loss_mask)
    else:
        raise Exception('nlp_model_type error!')


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    print_rank_0('> building train, validation, and test datasets '
                 f'for {args.nlp_model_type} ...')
    if args.nlp_model_type == 'gpt':
        kwargs = dict(
            seq_length=args.seq_length,
            train_data_prefix=args.train_data_path,
            valid_data_prefix=args.valid_data_path,
            test_data_prefix=args.test_data_path,
        )
    elif args.nlp_model_type == 'bert':
        kwargs = dict(
            seq_length=args.seq_length,
            masked_lm_prob=args.mask_prob,
            short_seq_prob=args.short_seq_prob,
            binary_head=args.bert_binary_head
        )
    elif args.nlp_model_type == 't5':
        kwargs = dict(
            max_seq_length=args.encoder_seq_length,
            max_seq_length_dec=args.decoder_seq_length,
            masked_lm_prob=args.mask_prob,
            short_seq_prob=args.short_seq_prob,
            dataset_type='t5'
        )
    else:
        raise Exception('nlp_model_type error!')

    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        **kwargs)
    print_rank_0(f"> finished creating {args.nlp_model_type} datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    init_mpi_env()
    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder, forward_step)
