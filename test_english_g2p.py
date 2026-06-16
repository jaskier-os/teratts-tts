from english_g2p import arpa_to_ru_symbols, english_to_ru_symbols


def test_arpa_vowel_stress_mapping():
    assert arpa_to_ru_symbols(["HH", "AH0", "L", "OW1"]) == ["h", "a0", "l", "o1", "u0"]


def test_arpa_consonant_clusters():
    assert arpa_to_ru_symbols(["S", "K", "EH1", "JH", "UH0", "L"]) == \
        ["s", "k", "e1", "d", "zh", "u0", "l"]


def test_th_and_w_substitution():
    assert arpa_to_ru_symbols(["TH", "AO1", "T"]) == ["s", "o1", "t"]
    assert arpa_to_ru_symbols(["W", "IY1", "K"]) == ["v", "i1", "k"]


def test_english_word_end_to_end():
    syms = english_to_ru_symbols("computer")
    assert syms == ["k", "a0", "m", "p", "j", "u1", "t", "e0", "r"]


def test_empty_and_nonalpha():
    assert arpa_to_ru_symbols([]) == []
    assert arpa_to_ru_symbols(["", "  "]) == []


from english_g2p import split_by_script


def test_split_pure_cyrillic():
    assert split_by_script("привет мир") == [("привет мир", "ru")]


def test_split_pure_latin():
    assert split_by_script("hello world") == [("hello world", "en")]


def test_split_mixed_attaches_spaces_and_punct():
    out = split_by_script("Я открыл Google и meeting")
    assert out == [("Я открыл ", "ru"), ("Google ", "en"), ("и ", "ru"), ("meeting", "en")]


def test_split_digits_attach():
    assert split_by_script("версия 5 build") == [("версия 5 ", "ru"), ("build", "en")]
