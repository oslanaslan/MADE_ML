"""Inverted Index Creator

Creates Inverted Index dictionary for query searching

"""

from __future__ import annotations, absolute_import

import os
import json
import re
import pickle
from collections import defaultdict
from typing import Dict, List


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
        elif method == 'binary':
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
        elif method == 'binary':
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


def main():
    documents = load_documents("/path/to/dataset")
    inverted_index = build_inverted_index(documents)
    inverted_index.dump("/path/to/inverted.index")
    inverted_index = InvertedIndex.load("/path/to/inverted.index")
    document_ids = inverted_index.query(["two", "words"])
    print(document_ids)


if __name__ == "__main__":
    main()
