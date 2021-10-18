"""
Tests for Inverted Index module

"""

import json
import pytest

from task_Ashabokov_Aslan_inverted_index_lib import InvertedIndex
from task_Ashabokov_Aslan_inverted_index_lib import load_documents
from task_Ashabokov_Aslan_inverted_index_lib import build_inverted_index

# toad_documents tests

def test_load_documents_invalid_path_name():
    """Test load_documents with invalid file path"""

    invalid_path = "some/invalid/path"
    with pytest.raises(FileNotFoundError):
        load_documents(invalid_path)


def test_load_documents_read_invalid_data(tmp_path):
    """Test load documents with invalid input data"""

    invalid_data = 'input string 1\n1 input_string 2\n\t3 input string 3\n\t4\tinput 5\n'
    invalid_data_path = tmp_path / "invalid_data.txt"

    with open(invalid_data_path, 'w') as file:
        file.write(invalid_data)

    with pytest.raises(ValueError):
        load_documents(invalid_data_path)

    invalid_data = "1\tsome text\na\tsome text\n"

    with open(invalid_data_path, 'w') as file:
        file.write(invalid_data)

    with pytest.raises(ValueError):
        load_documents(invalid_data_path)


def test_load_documents_read_valid_data(tmp_path):
    """Test load documents with valid input data"""

    valid_data = "1\tfirst text line\n2\tsecond text line\n3\tthird text line\n"
    valid_data_path = tmp_path / "valid_data.txt"
    valid_result = {
        1: "first text line",
        2: "second text line",
        3: "third text line"
    }

    with open(valid_data_path, 'w') as file:
        file.write(valid_data)

    result = load_documents(valid_data_path)

    assert result == valid_result, (
        f"Result must be: {valid_result}\nGot: {result}\n"
    )

# build_inverted_index tests

def test_build_inverted_index_correct_result_type():
    """Test build if it creates InvertedIndex"""

    data_dict = {
        1: "first text line",
        2: "second text line",
        3: "third text line"
    }
    result = build_inverted_index(data_dict)

    assert isinstance(result, InvertedIndex), (
        f"result must be type of {type(InvertedIndex)}\nGot: {type(result)}"
    )

def test_build_inverted_index_empty_input():
    """Test build with empty input dict"""

    empty_dict = {}
    result = build_inverted_index(empty_dict)

    assert result.data_ == empty_dict, (
        f"result.data_ must be empty dict\nGot: {result.data_}"
    )

def test_build_inverted_index_invalid_input():
    """Test build with invalid input data"""

    invalid_data = {
        '1': 'aa',
        '2': 'asdcd',
        '3': "wefwer"
    }

    with pytest.raises(ValueError):
        build_inverted_index(invalid_data)

    invalid_data = {
        1: 1,
        2: 2,
        3: 3
    }

    with pytest.raises(ValueError):
        build_inverted_index(invalid_data)

def test_build_inverted_index_valid_input():
    """Test build with valid input data"""
    valid_input = {
        1: "first text line",
        2: "second text line",
        3: "third text line"
    }
    valid_output = {
        'first': [1],
        'text': [1, 2, 3],
        'line': [1, 2, 3],
        'second': [2],
        'third': [3]
    }
    result = build_inverted_index(valid_input)
    is_result_valid = True

    for key in result.data_:
        if key not in valid_output:
            is_result_valid = False

    for key in valid_output:
        if key not in result.data_:
            is_result_valid = False

    if is_result_valid:
        for key in valid_output:
            valid_lst = valid_output[key]
            result_lst = result.data_[key]

            for item in valid_lst:
                if item not in result_lst:
                    is_result_valid = False

            for item in result_lst:
                if item not in valid_lst:
                    is_result_valid = False

    assert is_result_valid, (
        f"result.data_ must be: {valid_output}\nGot: {result.data_}\n"
    )

# InvertedIndex.__init__ test

def test_init_type_error():
    """Test InvertedIndex init with wrong type input"""

    with pytest.raises(TypeError):
        InvertedIndex("dict")

def test_init_value_error():
    """Test InvertedIndex init with invalid input data"""

    invalid_dict = {
        1: "asad",
        2: "asdasedc"
    }

    with pytest.raises(ValueError):
        InvertedIndex(invalid_dict)

    invalid_dict = {
        "asdasdf": 1,
        'sdfdsf': 4
    }

    with pytest.raises(ValueError):
        InvertedIndex(invalid_dict)

# InvertedIndex.dump tests

def test_dump_invalid_path_name():
    """Test dump with invalid path"""

    file_path = "some/invalid/path"
    valid_dict = {
        'a': [1],
        'b': [2, 3]
    }
    inverted_index = InvertedIndex(valid_dict)

    with pytest.raises(ValueError):
        inverted_index.dump(file_path)

def test_dump_valid_data(tmp_path):
    """Test dump with valid data"""

    file_path = tmp_path / "inversed.index"
    valid_dict = {
        'a': [1],
        'b': [2, 3]
    }
    inverted_index = InvertedIndex(valid_dict)
    inverted_index.dump(file_path)

# InvertedIndex.load tests

def test_load_invalid_path_name():
    """Test load with invalid path"""

    invalid_path = "some/invalid/path"
    with pytest.raises(FileNotFoundError):
        InvertedIndex.load(invalid_path)

def test_load_read_invalid_json(tmp_path):
    """Test load with invalid json"""

    file_path = tmp_path / "invalid.json"
    invalid_data = "sdjfkjsd"

    with open(file_path, 'w') as file:
        file.write(invalid_data)

    with pytest.raises(ValueError):
        InvertedIndex.load(file_path)

def test_load_read_empty_dict(tmp_path):
    """Test load with empy json"""

    empty_dict_path = tmp_path / "empty_dict.json"
    empty_dict_path.touch()
    with open(empty_dict_path, 'w') as file:
        json.dump({}, file)

    result = InvertedIndex.load(empty_dict_path)

    assert len(result.data_) == 0, (
        f"result must be empty dict. Got: {result.data_}\n"
    )

def test_load_read_invalid_dict(tmp_path):
    """Test load with invalid data"""

    invalid_data_path = tmp_path / "invalid_dict.json"
    invalid_data_path.touch()
    invalid_dict = {
        'a': ['a', 'b', 'c'],
        'b': None
    }

    with open(invalid_data_path, 'w') as file:
        json.dump(invalid_dict, file)

    with pytest.raises(Exception):
        InvertedIndex.load(invalid_data_path)


def test_load_result_type(tmp_path):
    """Test load function output"""

    valid_data_path = tmp_path / "valid_dict.json"
    valid_data = {
        'a': [1, 2, 3],
        'abc': [5, 1, 19]
    }

    with open(valid_data_path, 'w') as file:
        json.dump(valid_data, file)

    result = InvertedIndex.load(valid_data_path)

    assert isinstance(result, InvertedIndex), (
        f"Result must be type of InvertedIndex\nGot: {type(result)}\n"
    )

def test_load_read_valid_data(tmp_path):
    """Test load with valid data"""

    valid_data_path = tmp_path / "valid_dict.json"
    valid_data = {
        'a': [1, 2, 3],
        'abc': [5, 1, 19]
    }

    with open(valid_data_path, 'w') as file:
        json.dump(valid_data, file)

    result = InvertedIndex.load(valid_data_path)

    assert valid_data == result.data_, (
        f"Result.data_ must be: {valid_data}\nGot: {result.data_}"
    )

# InvertedIndex.query tests

def test_query_invalid_input():
    """Test query with invalid input"""

    valid_data = {
        'a': [1, 2, 3],
        'abc': [5, 1, 19]
    }
    inverted_index = InvertedIndex(valid_data)
    invalid_input = "ewdfewf"

    with pytest.raises(TypeError):
        inverted_index.query(invalid_input)

    invalid_input = [1, 2, 3, None]

    with pytest.raises(TypeError):
        inverted_index.query(invalid_input)

    invalid_input = ['a', 'b', None]

    with pytest.raises(TypeError):
        inverted_index.query(invalid_input)

def test_query_valid_input():
    """Test query with valid input data"""

    valid_data = {
        'a': [1, 2, 3],
        'abc': [5, 1, 2],
        'c': [2, 6, 3]
    }
    inverted_index = InvertedIndex(valid_data)

    assert  len(set(inverted_index.query([]))) == 0
    assert len(set([1, 2, 3]).symmetric_difference(inverted_index.query(['a']))) == 0
    assert len(set([1, 2]).symmetric_difference(inverted_index.query(['a', 'abc']))) == 0
    assert len(set([2, 3]).symmetric_difference(inverted_index.query(['a', 'c']))) == 0
    assert len(set([2]).symmetric_difference(inverted_index.query(['abc', 'c', 'a']))) == 0


# main tests

# def test_main_test():
#     """Test main function"""

#     main()

def test_full():
    """Full test"""

    filename = "sample.txt"
    documents = load_documents(filename)
    inverted_index = build_inverted_index(documents)
    inverted_index.dump("inverted.index")
    inverted_index = InvertedIndex.load("inverted.index")
    document_ids = inverted_index.query(["two", "words"])
    result = set()
    assert len(result.symmetric_difference(set(document_ids))) == 0, (
        "full test 1 failed\n"
    )
    document_ids = inverted_index.query(["text", "line"])
    result = set([5, 7])
    assert len(result.symmetric_difference(document_ids)) == 0, (
        "full test 2 failed\n"
    )
    document_ids = inverted_index.query(["something", "else"])
    result = set([78])
    assert len(result.symmetric_difference(document_ids)) == 0, (
        "full test 3 failed\n"
    )
    document_ids = inverted_index.query(["something", "ggvp"])
    result = set([])
    assert len(result.symmetric_difference(document_ids)) == 0, (
        "full test 4 failed\n"
    )
