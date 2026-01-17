#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for NeoN input system Python bindings."""

import pytest
import neon


# TokenList Tests
def test_token_list_construction_empty():
    """Test empty TokenList construction."""
    token_list = neon.TokenList()
    assert token_list.empty()
    assert token_list.size() == 0


def test_token_list_insert_operations():
    """Test insert operations."""
    token_list = neon.TokenList()

    # Insert different types
    token_list.insert_int(42)
    token_list.insert_double(3.14)
    token_list.insert_string("hello")
    token_list.insert_bool(True)

    assert token_list.size() == 4
    assert not token_list.empty()


def test_token_list_basic_operations():
    """Test basic TokenList operations."""
    token_list = neon.TokenList()

    # Insert some values
    token_list.insert_int(1)
    token_list.insert_int(2)
    token_list.insert_int(3)

    assert token_list.size() == 3

    # Remove middle element
    token_list.remove(1)
    assert token_list.size() == 2


def test_token_list_get_int():
    """Test typed get method for integers."""
    token_list = neon.TokenList()
    token_list.insert_int(42)
    token_list.insert_int(100)

    value1 = token_list.get_int(0)
    value2 = token_list.get_int(1)

    assert value1 == 42
    assert value2 == 100


def test_token_list_get_double():
    """Test typed get method for doubles."""
    token_list = neon.TokenList()
    token_list.insert_double(3.14)
    token_list.insert_double(2.71)

    value1 = token_list.get_double(0)
    value2 = token_list.get_double(1)

    assert abs(value1 - 3.14) < 1e-10
    assert abs(value2 - 2.71) < 1e-10


def test_token_list_get_string():
    """Test typed get method for strings."""
    token_list = neon.TokenList()
    token_list.insert_string("hello")
    token_list.insert_string("world")

    value1 = token_list.get_string(0)
    value2 = token_list.get_string(1)

    assert value1 == "hello"
    assert value2 == "world"


def test_token_list_get_bool():
    """Test typed get method for booleans."""
    token_list = neon.TokenList()
    token_list.insert_bool(True)
    token_list.insert_bool(False)

    value1 = token_list.get_bool(0)
    value2 = token_list.get_bool(1)

    assert value1 is True
    assert value2 is False


def test_token_list_repr():
    """Test string representation."""
    token_list = neon.TokenList()
    repr_str = repr(token_list)
    assert "TokenList" in repr_str
    assert "size=0" in repr_str


# Dictionary Tests
def test_dictionary_construction_empty():
    """Test empty Dictionary construction."""
    dictionary = neon.Dictionary()
    assert dictionary.empty()
    assert dictionary.size() == 0


def test_dictionary_insert_operations():
    """Test insert operations."""
    dictionary = neon.Dictionary()

    # Insert different types
    dictionary.insert_int("age", 25)
    dictionary.insert_double("pi", 3.14159)
    dictionary.insert_string("name", "NeoN")
    dictionary.insert_bool("active", True)

    assert dictionary.size() == 4
    assert not dictionary.empty()


def test_dictionary_contains_operation():
    """Test contains operation."""
    dictionary = neon.Dictionary()
    dictionary.insert_int("key1", 42)
    dictionary.insert_string("key2", "value")

    assert dictionary.contains("key1")
    assert dictionary.contains("key2")
    assert not dictionary.contains("nonexistent")


def test_dictionary_remove_operation():
    """Test remove operation."""
    dictionary = neon.Dictionary()
    dictionary.insert_int("key1", 42)
    dictionary.insert_string("key2", "value")

    assert dictionary.size() == 2
    dictionary.remove("key1")
    assert dictionary.size() == 1
    assert not dictionary.contains("key1")
    assert dictionary.contains("key2")


def test_dictionary_typed_get_methods():
    """Test typed get methods."""
    dictionary = neon.Dictionary()
    dictionary.insert_int("age", 30)
    dictionary.insert_double("height", 175.5)
    dictionary.insert_string("name", "Alice")
    dictionary.insert_bool("married", False)

    assert dictionary.get_int("age") == 30
    assert abs(dictionary.get_double("height") - 175.5) < 1e-10
    assert dictionary.get_string("name") == "Alice"
    assert dictionary.get_bool("married") is False


def test_dictionary_keys_method():
    """Test keys() method."""
    dictionary = neon.Dictionary()
    dictionary.insert_int("key1", 1)
    dictionary.insert_string("key2", "value")
    dictionary.insert_bool("key3", True)

    keys = dictionary.keys()
    assert len(keys) == 3
    assert "key1" in keys
    assert "key2" in keys
    assert "key3" in keys


def test_dictionary_repr():
    """Test string representation."""
    dictionary = neon.Dictionary()
    repr_str = repr(dictionary)
    assert "Dictionary" in repr_str
    assert "size=0" in repr_str


# Input Tests
def test_input_construction_from_dictionary():
    """Test Input construction from Dictionary."""
    dictionary = neon.Dictionary()
    dictionary.insert_string("test", "value")

    input_obj = neon.Input(dictionary)
    assert input_obj.is_dictionary()
    assert not input_obj.is_token_list()


def test_input_construction_from_token_list():
    """Test Input construction from TokenList."""
    token_list = neon.TokenList()
    token_list.insert_int(42)

    input_obj = neon.Input(token_list)
    assert input_obj.is_token_list()
    assert not input_obj.is_dictionary()


def test_input_type_checking():
    """Test type checking methods."""
    # Test with Dictionary
    dict_input = neon.Input(neon.Dictionary())
    assert dict_input.is_dictionary()
    assert not dict_input.is_token_list()

    # Test with TokenList
    list_input = neon.Input(neon.TokenList())
    assert list_input.is_token_list()
    assert not list_input.is_dictionary()


def test_input_safe_access_dictionary():
    """Test safe access to Dictionary variant."""
    dictionary = neon.Dictionary()
    dictionary.insert_string("key", "value")

    input_obj = neon.Input(dictionary)
    retrieved_dict = input_obj.get_dictionary()

    assert retrieved_dict.contains("key")
    assert retrieved_dict.get_string("key") == "value"


def test_input_safe_access_token_list():
    """Test safe access to TokenList variant."""
    token_list = neon.TokenList()
    token_list.insert_int(42)

    input_obj = neon.Input(token_list)
    retrieved_list = input_obj.get_token_list()

    assert retrieved_list.size() == 1
    assert retrieved_list.get_int(0) == 42


def test_input_repr():
    """Test string representation."""
    dict_input = neon.Input(neon.Dictionary())
    list_input = neon.Input(neon.TokenList())

    dict_repr = repr(dict_input)
    list_repr = repr(list_input)

    assert "Input containing Dictionary" in dict_repr
    assert "Input containing TokenList" in list_repr
