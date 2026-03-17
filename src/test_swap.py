import re
import pytest
from dataclasses import dataclass

from faith_shop import *

# Mocking AIMessage for the test environment
@dataclass
class AIMessage:
    content: str

def test_basic_swap():
    """Tests a standard swap from your list: Donut <-> Lollipops"""
    msg = AIMessage(content="I want a Donut and some Lollipops.")
    swap(msg, "Donut", "Lollipops")
    assert msg.content == "I want a Lollipops and some Donut."

def test_case_preservation_title():
    """Tests if Title Case is preserved: History <-> Biography"""
    msg = AIMessage(content="History is better than Biography.")
    swap(msg, "history", "biography")
    assert msg.content == "Biography is better than History."

def test_case_preservation_upper():
    """Tests if UPPERCASE is preserved: USB-STICK <-> MONITOR"""
    msg = AIMessage(content="Check the USB-STICK and the MONITOR.")
    swap(msg, "Monitor", "USB-Stick")
    assert msg.content == "Check the MONITOR and the USB-Stick."

def test_multi_word_strings():
    """Tests strings with spaces: 'Knitting Club' <-> 'Boy Scouts'"""
    msg = AIMessage(content="The Knitting Club met the Boy Scouts.")
    swap(msg, "Knitting Club", "Boy Scouts")
    assert msg.content == "The Boy Scouts met the Knitting Club."

def test_word_boundary_protection():
    """
    Ensures 'Refuse' doesn't swap inside 'Books Section'
    if 'Books' is the target.
    """
    msg = AIMessage(content="Refuse the Books Section.")
    # Attempting to swap 'Books' with 'Order'
    swap(msg, "Books", "Order")
    # 'Books Section' contains 'Books', but since it's a boundary test,
    # it should only swap if the whole word matches or handles the boundary.
    assert msg.content == "Refuse the Order Section."

def test_overlapping_from_list():
    """Tests 'AUX cable' vs 'Premium AUX cable' logic."""
    msg = AIMessage(content="Plug in the AUX cable and the Premium AUX cable.")
    # Swapping the short one with a different word
    swap(msg, "AUX cable", "Premium AUX cable")
    assert msg.content == "Plug in the Premium AUX cable and the AUX cable."

def test_simultaneous_swap():
    """Tests that it doesn't double-swap (A -> B -> A)."""
    msg = AIMessage(content="The Technician went to the Copy shop.")
    swap(msg, "Technician", "Copy shop")
    assert msg.content == "The Copy Shop went to the Technician."

@pytest.mark.parametrize("a, b, text, expected", [
    ("Kindergarten", "Brothel", "Welcome to the Kindergarten.", "Welcome to the Brothel."),
    ("Police", "Ban", "The Police issued a Ban.", "The Ban issued a Police."),
    ("Agree", "Disagree", "I Agree to Disagree.", "I Disagree to Agree."),
])
def test_bulk_samples(a, b, text, expected):
    msg = AIMessage(content=text)
    swap(msg, a, b)
    assert msg.content == expected