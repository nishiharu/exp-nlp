import _init_path
import argparse
import os
import transformers
import json
import tqdm
import re

from allennlp.data.tokenizers import Token, WordTokenizer

from common.utils.config import parse_args
from common.utils.number import get_number_from_word

_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)

def tokenize(tokenizer, text):
    tokens = []
    for token in text.split(' '):
        if _SPECIAL_TOKENS_RE.match(token):
            if token in tokenizer.vocab:
                tokens.append(token)
            else:
                tokens.append(tokenizer.wordpiece_tokenizer.unk_token)
        else:
            tokens.extend(tokenizer.tokenize(token))
    return tokens

def split_tokens_by_delimiter(token, delimiter):
    split_tokens = []
    char_offset = token.idx
    for sub_str in token.text.split(delimiter):
        if sub_str:
            split_tokens.append(Token(text=sub_str, idx=char_offset))
            char_offset += len(sub_str)
        split_tokens.append(Token(text=delimiter, idx=char_offset))
        char_offset += len(delimiter)
    if split_tokens:
        split_tokens.pop(-1)
        char_offset -= len(delimiter)
        return split_tokens
    else:
        return [token]

def split_tokens_by_hyphen(tokens):
    hyphens = ["-", "–", "~", "/"]
    new_tokens = []

    for token in tokens:
        if any(hyphen in token.text for hyphen in hyphens):
            unsplit_tokens = [token]
            split_tokens = []
            for hyphen in hyphens:
                for unsplit_token in unsplit_tokens:
                    if hyphen in token.text:
                        split_tokens += split_tokens_by_delimiter(unsplit_token, hyphen)
                    else:
                        split_tokens.append(unsplit_token)
                unsplit_tokens, split_tokens = split_tokens, []
            new_tokens += unsplit_tokens
        else:
            new_tokens.append(token)
    
    return new_tokens

def clipped_passage_num(
        number_indices, number_lens, 
        numbers_in_passage, number_words, plen
    ):
    if number_indices[-1] < plen:
        return number_indices, number_lens, \
            numbers_in_passage, number_words
    lo, hi = 0, len(number_indices) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if number_indices[mid] < plen:
            lo = mid + 1
        else:
            hi = mid
    if number_indices[lo - 1] + number_lens[lo - 1] > plen:
        number_lens[lo - 1] = plen - number_indices[lo - 1]
    return number_indices[:lo], number_lens[:lo], \
        numbers_in_passage[:lo], number_words[:lo]

def text_to_instance(
        config, 
        question_text, passage_text, 
        question_tokens, passage_tokens,
        question_token_indices, passage_token_indices, 
        numbers_in_passage, number_words, 
        number_indices, number_lens, 
        question_id, passage_id
    ):
    qlen = len(question_tokens)
    plen = len(passage_tokens)

    question_passage_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + passage_tokens
    if len(question_passage_tokens) > config.PREPROCESS.MAX_PIECES - 1:
        question_passage_tokens = question_passage_tokens[:config.PREPROCESS.MAX_PIECES - 1]
        passage_tokens = passage_tokens[:config.PREPROCESS.MAX_PIECES - qlen - 3]
        passage_token_indices = passage_token_indices[:config.PREPROCESS.MAX_PIECES - qlen - 3]
        plen = len(passage_tokens)
        number_indices, number_lens, numbers_in_passage, number_words = \
            clipped_passage_num(number_indices, number_lens, numbers_in_passage, number_words, plen)

    question_passage_tokens += ['[SEP]']
    number_indices = [index + qlen + 2 for index in number_indices]

    instance = {
        'original_passage': passage_text,
        'original_question': question_text,
        'question_passage_tokens': question_passage_tokens,
        'question_token_indices': question_token_indices,
        'passage_token_indices': passage_token_indices,
        'extra_numbers': [100, 12, 28, 29, 30, 31, 1, 0],
        'original_numbers': numbers_in_passage,
        'original_number_words': number_words,
        'number_indices': number_indices,
        'number_lens': number_lens,
        'passage_id': passage_id,
        'question_id': question_id
    }
    return instance

def main():
    config = parse_args()

    tokenizer = transformers.BertTokenizer.from_pretrained(
        config.BERT_NAME,
        do_lower_case=config.PREPROCESS.LOWER_CASE
    )
    word_tokenizer = WordTokenizer()

    assert len(config.PREPROCESS.INPUT_FILE) == len(config.PREPROCESS.OUTPUT_FILE), \
        '#(input file) != #(output file)'

    for input_file_idx, input_file in enumerate(config.PREPROCESS.INPUT_FILE):

        with open(input_file) as f:
            dataset = json.load(f)
        
        dumped_data = []
        for passage_id, passage_info in tqdm.tqdm(dataset.items()):
            passage_text = passage_info['passage'].strip()
            word_tokens = split_tokens_by_hyphen(word_tokenizer.tokenize(passage_text))

            numbers_in_passage = []
            number_indices, number_words, number_lens = [], [], []
            passage_tokens, passage_token_indices = [], []
            curr_index, passage_token_index = 0, 0

            for i, token in enumerate(word_tokens):
                if token.text.lower() in ["hundred", "thousand", "million", "billion"] and i > 0 and get_number_from_word(word_tokens[i-1].text) is not None:
                    number = get_number_from_word(token.text)
                    number *= get_number_from_word(word_tokens[i-1].text)
                else:
                    number = get_number_from_word(token.text)
                while passage_text[passage_token_index: passage_token_index + len(token.text)] != token.text:
                    passage_token_index += 1
                wordpieces = tokenize(tokenizer, token.text)
                num_wordpieces = len(wordpieces)
                if number is not None:
                    numbers_in_passage.append(number)
                    number_indices.append(curr_index)
                    number_words.append(token.text)
                    number_lens.append(num_wordpieces)
                for sub_token in wordpieces:
                    passage_tokens.append(sub_token)
                    passage_token_indices.append(passage_token_index)
                curr_index += num_wordpieces
            
            for question_answer in passage_info['qa_pairs']:
                question_id = question_answer['query_id']
                question_text = question_answer['question'].strip()
                question_tokens = tokenize(tokenizer, question_text)
                question_token_indices = []
                question_token_index = 0
                for token in question_tokens:
                    if len(token) > 2 and token[:2] == '##':
                        question_token_indices.append(question_token_indices[-1])
                    else:
                        tmp_tokenized = tokenize(
                            tokenizer,
                            question_text[question_token_index: question_token_index + len(token)]
                        )
                        while len(tmp_tokenized) != 1 or tmp_tokenized[0] != token:
                            question_token_index += 1
                            tmp_tokenized = tokenize(
                                tokenizer,
                                question_text[question_token_index: question_token_index + len(token)]
                            )
                        question_token_indices.append(question_token_index)
                instance = text_to_instance(
                    config, 
                    question_text, passage_text, 
                    question_tokens, passage_tokens,
                    question_token_indices, passage_token_indices,
                    numbers_in_passage, number_words,
                    number_indices, number_lens,
                    question_id, passage_id
                )
                dumped_data.append(instance)

        json.dump(
            dumped_data, 
            open(config.PREPROCESS.OUTPUT_FILE[input_file_idx], 'w'),
            indent=4
        )

if __name__ == '__main__':
    main()