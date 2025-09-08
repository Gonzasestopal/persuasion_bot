import pytest

from app.domain.errors import InvalidContinuationMessage, InvalidStartMessage
from app.domain.parser import assert_no_topic_or_side_markers, parse_topic_side


def test_parser_empty_message():
    with pytest.raises(InvalidStartMessage, match='must not be empty'):
        parse_topic_side('')


def test_parser_missing_topic():
    with pytest.raises(InvalidStartMessage, match='topic is missing'):
        parse_topic_side('Side: PRO')


def test_parser_missing_side():
    with pytest.raises(InvalidStartMessage, match='side is missing'):
        parse_topic_side('Topic: Dogs are great')


def test_parser_unsupported_side():
    with pytest.raises(InvalidStartMessage, match="must be 'pro' or 'con'"):
        parse_topic_side('Topic: Dogs are great, Side: maybe')


def test_parser_valid_side_and_topic_mixed_case():
    topic = 'Dogs are humans best friend'
    t, s = parse_topic_side(f'Topic: {topic} , Side: Pro')
    assert t == topic
    assert s == 'pro'


def test_parser_uppercase_input():
    topic = 'DOGS ARE HUMANS BEST FRIENDS'
    t, s = parse_topic_side(f'TOPIC: {topic}, SIDE: PRO')
    assert t == topic
    assert s == 'pro'


def test_parser_lowercase_input():
    topic = 'dogs are human best friends'
    t, s = parse_topic_side(f'topic: {topic}, side: pro')
    assert t == topic
    assert s == 'pro'


def test_parser_reversed_order_and_punctuation():
    topic = 'Dogs are human best friends'
    t, s = parse_topic_side(f'Side: CON. Topic: {topic}')
    assert t == topic
    assert s == 'con'
    assert t == topic
    assert s == 'con'
    assert s == 'con'


def test_parset_check_maximum_length_input():
    topic = 'Topic: This is a simple example string that contains exactly one hundred one characters in total length now'
    side = ' Side: pro'
    with pytest.raises(InvalidStartMessage, match='must be less than 100 characters'):
        parse_topic_side(topic + side)


def test_parser_rejects_message_over_160_chars():
    # Construimos un mensaje de más de 120 caracteres
    long_message = 'a' * 160
    with pytest.raises(
        InvalidContinuationMessage, match='must be less than 160 characters'
    ):
        assert_no_topic_or_side_markers(long_message)


def test_parser_accepts_message_under_160_chars():
    # Construimos un mensaje de menos de 160 caracteres
    message = 'I think dogs are better than cats because they provide partnership.'
    # No debe lanzar excepción
    assert assert_no_topic_or_side_markers(message)
