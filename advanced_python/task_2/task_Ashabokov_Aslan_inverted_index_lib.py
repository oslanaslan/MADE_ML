#!/usr/bin/env python
"""Inverted Index Creator

Creates Inverted Index dictionary for query searching

"""

from __future__ import annotations, absolute_import

import os
import sys
import json
import re
import pickle
from collections import defaultdict
from typing import Dict, List
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


DEFAULT_OUTPUT_FILENAME = "inverted.index"


class InvertedIndex:
    """A class for searching through inverted index

    Attributes
    ----------
    data_: dict
        contains dict of inverted indices

    Methods
    -------
    dict_check(dict)
        raises error if dict is not valid
    query(list)
        returns a list with indices
    dump(filepath)
        dumps json to file
    load(filepath)
        loads json from file
    """

    def __init__(self, inverted_index_dict: Dict[int, str] = None) -> None:
        inverted_index_dict = inverted_index_dict or dict()
        self.dict_check(inverted_index_dict)
        self.data_ = defaultdict(list, inverted_index_dict)

    def __eq__(self, other: InvertedIndex) -> bool:
        if not isinstance(other, InvertedIndex):
            raise NotImplementedError(
                f"InvertedIndex.__eq__() not implemented for objects type of {type(other)}"
            )

        if set(self.data_.keys()) != set(other.data_.keys()):
            return False

        is_eq = True

        for key in self.data_:
            if set(self.data_[key]) != set(other.data_[key]):
                is_eq = False
                break

        return is_eq

    def __ne__(self, other: InvertedIndex) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return f"InvertedIndex: {self.data_.__repr__()}"

    def __str__(self) -> str:
        return self.__repr__()

    @staticmethod
    def dict_check(input_dict: dict):
        """Checks if input_dict is a valid inverted index dict"""

        if not isinstance(input_dict, dict):
            raise TypeError(f"inverted_index_dict must be type of {type(dict)}\n\
                Got: {type(input_dict)}")

        for key in input_dict:
            if not isinstance(key, str):
                raise ValueError(f"dict keys must be type of {type(int)}\n\
                    Got: {type(input_dict)}")

            item_lst = input_dict[key]

            if not isinstance(item_lst, list):
                raise ValueError(f"dict values must be typeof {type(list)}\n\
                    Got: {type(item_lst)}")

            for item in item_lst:
                if not isinstance(item, int):
                    raise ValueError(f"dict must be list of elements type of {type(int)}\n\
                        Got: {type(item)}")

    def query(self, words: List[str]) -> List[int]:
        """Return the list of relevant documents for the given query"""

        if not isinstance(words, list):
            raise TypeError(f"words must type of {type(list)}\n\
                Got: {type(words)}")

        for item in words:
            if not isinstance(item, str):
                raise TypeError(f"items of words must be type of {type(str)}\n\
                    Got: {type(item)}")

        if len(self.data_) == 0 or len(words) == 0:
            return []

        words = [word.lower().strip() for word in words]
        result = set(self.data_[words[0]])

        for word in words:
            result = result.intersection(set(self.data_[word]))

        result = list(result)

        return result

    def dump(self, filepath: str, method: str = 'json') -> None:
        """Dumps self.data_ to filepath in json format"""

        filepath = str(filepath)

        if method == 'json':
            JsonPolicy.dump(self.data_, filepath)
        elif method == 'pickle':
            PicklePolicy.dump(self.data_, filepath)
        else:
            raise NotImplementedError(f"InvertedIndex.dump: not implemented\
                for method={method}\n")

    @classmethod
    def load(cls, filepath: str, method: str = 'json') -> InvertedIndex:
        """Loads self.data_ from file in json format"""

        filepath = str(filepath)

        if method == 'json':
            inverted_index = JsonPolicy.load(filepath)
        elif method == 'pickle':
            inverted_index = PicklePolicy.load(filepath)
        else:
            raise NotImplementedError(f"InvertedIndex.load: not implemented\
                for method={method}\n")

        return inverted_index


def load_documents(filepath: str) -> Dict[int, str]:
    """Loads strings and row numbers from file with text"""

    filepath = str(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError("load_documents: filepath doesn't exist\n")

    with open(filepath, 'r') as file:
        lines = file.readlines()

    data_dict = {}

    for line in lines:
        if '\t' not in line:
            raise ValueError("load_documents: got invalid data\n")

        splited_line = line.lower().split('\t', maxsplit=1)

        try:
            index = int(splited_line[0])
        except:
            raise ValueError("load_documents: got invalid data\n")

        data_dict[index] = splited_line[1].strip()

    return data_dict


def build_inverted_index(documents: Dict[int, str]) -> InvertedIndex:
    """Builds inverted index dict based on data, loaded with load_documents"""

    inverted_index_dict = defaultdict(list)

    for index in documents:
        if not isinstance(index, int):
            raise ValueError(f"Document index must be int\nGot: {type(index)}")
        if not isinstance(documents[index], str):
            raise ValueError(f"Document item must be str\nGot: {type(documents[index])}\n")
        item_lst = documents[index].strip()
        item_lst = re.split(r"\W+", item_lst)
        item_lst = [item.strip() for item in item_lst]

        for word in item_lst:
            inverted_index_dict[word].append(index)

    inverted_index = InvertedIndex(inverted_index_dict)

    return inverted_index


class StoragePolicy:
    @staticmethod
    def dump(word_to_docs_mapping: dict, filepath: str):
        pass

    @staticmethod
    def load(filepath: str):
        pass


class JsonPolicy(StoragePolicy):
    @staticmethod
    def dump(word_to_docs_mapping: dict, filepath: str):
        dirname = os.path.dirname(filepath)

        if not os.path.isdir(dirname) and not dirname == '':
            raise FileNotFoundError(f"Dirpath {filepath} doesn't exist\n")

        InvertedIndex.dict_check(word_to_docs_mapping)

        with open(filepath, 'w') as file:
            json.dump(word_to_docs_mapping, file)

    @staticmethod
    def load(filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} doesn't exist\n")

        with open(filepath, 'r') as file:
            data_dict = json.load(file)

        return InvertedIndex(data_dict)


class PicklePolicy(StoragePolicy):
    @staticmethod
    def dump(word_to_docs_mapping: dict, filepath: str):
        dirname = os.path.dirname(filepath)

        if not os.path.isdir(dirname) and not dirname == '':
            raise FileNotFoundError(f"Dirpath {filepath} doesn't exist\n")

        InvertedIndex.dict_check(word_to_docs_mapping)

        with open(filepath, 'wb') as file:
            pickle.dump(word_to_docs_mapping, file)

    @staticmethod
    def load(filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dirpath {filepath} doesn't exist\n")

        with open(filepath, 'rb') as file:
            data_dict = pickle.load(file)

        return InvertedIndex(data_dict)

def setup_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="inverted_index",
        description="Inverted Index CLI: lib for loading, storing and processing inverted index mappings",
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=True,
        argument_default="-h",
    )
    subparsers = parser.add_subparsers(
        title="sommands",
        dest="command",
        required=True,
    )
    subparsers.required = True

    build_parser = subparsers.add_parser(
        "build",
        help="Inverted Index CLI: build: build inverted index",
    )
    build_parser.add_argument(
        "--strategy",
        help="dump strategy: dump to json or dump to pickle (default: json)",
        choices=["json", "pickle"],
        default="json",
    )
    build_parser.add_argument(
        "--dataset",
        help="path to dataset",
        dest="dataset_filename",
        required=True,
    )
    build_parser.add_argument(
        "--output",
        help="output file name (default: inverted.index)",
        default=DEFAULT_OUTPUT_FILENAME,
        dest="output_filename"
    )

    query_parser = subparsers.add_parser(
        "query",
        help="Inverted Index CLI: query: search for query result in inverted index",
    )
    query_parser.add_argument(
        "--json-index",
        help="Load Inverted Index CLI from json",
        dest="json_filename",
    )
    query_parser.add_argument(
        "--pickle-index",
        help="Load Inverted Index CLI from pickle",
        dest="pickle_filename",
    )
    query_parser.add_argument(
        "--query",
        action='append',
        required=True,
        nargs='+',
        dest="word",
    )

    return parser

def parse_arguments(parser: ArgumentParser) -> list:
    try:
        args = parser.parse_args()
    except:
        # parser.print_help(sys.stderr)
        sys.exit(0)

    return args

def main():
    parser = setup_parser()
    args = parse_arguments(parser)
    
    if args.command == "build":
        documents = load_documents(args.dataset_filename)
        inverted_index = build_inverted_index(documents)
        inverted_index.dump(args.output_filename, method=args.strategy)
    elif args.command == "query":
        inverted_index = InvertedIndex.load(args.)
    document_ids = inverted_index.query(["two", "words"])
    print(document_ids)


if __name__ == "__main__":
    main()
