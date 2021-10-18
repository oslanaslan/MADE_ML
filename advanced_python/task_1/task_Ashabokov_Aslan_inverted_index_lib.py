"""
Inverted Index Creator

Creates Inverted Index dictionary for query searching
"""

from __future__ import annotations, absolute_import

import os
from collections import defaultdict
import json
from typing import Dict, List
import re


class InvertedIndex:
    """
    A class for searching through inverted index

    Attributes
    ----------
    data_: dict
        contains dict of inverted indices

    Methods
    -------
    dict_check_(dict)
        raises error if dict is not valid
    query(list)
        returns a list with indices
    dump(filepath)
        dumps json to file
    load(filepath)
        loads json from file
    """

    def __init__(self, inverted_index_dict: Dict[int, str] = None) -> None:
        if not inverted_index_dict:
            inverted_index_dict = {}
        self.dict_check_(inverted_index_dict)
        self.data_ = inverted_index_dict

    @classmethod
    def dict_check_(cls, input_dict: dict):
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

        if len(words) == 0:
            return []

        result = set(self.data_[words[0]])

        for word in words:
            result = result.intersection(set(self.data_[word]))

        result = sorted(list(result))
        return result

    def dump(self, filepath: str) -> None:
        """Dumps self.data_ to filepath in json format"""

        filepath = str(filepath)

        try:
            with open(filepath, 'w') as file:
                json.dump(self.data_, file)
        except:
            raise ValueError(f"Path {filepath} doesn't exist\n")

    @classmethod
    def load(cls, filepath: str) -> InvertedIndex:
        """Loads self.data_ from file in json format"""

        filepath = str(filepath)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found\n")

        try:
            with open(filepath, 'r') as file:
                data_dict = json.load(file)
        except:
            raise ValueError(f"Can't read JSON from file {filepath}")

        cls.dict_check_(data_dict)
        return InvertedIndex(data_dict)


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

    inverted_index = InvertedIndex(dict(inverted_index_dict))
    return inverted_index


def main() -> None:
    """Main function"""

    documents = load_documents("wikipedia_sample")
    inverted_index = build_inverted_index(documents)
    inverted_index.dump("inverted.index")
    inverted_index = InvertedIndex.load("inverted.index")
    document_ids = inverted_index.query(["two", "words"])

    return document_ids


if __name__ == "__main__":
    main()
