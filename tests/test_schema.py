"""
Tests for response schema compliance.

Every response from the server must validate against the exact schema:
{
  "reply": "string",
  "recommendations": [{"name": "string", "url": "string", "test_type": "string"}],
  "end_of_conversation": false
}

- recommendations is [] (empty array, never null, never omitted) while clarifying/refusing
- recommendations has 1-10 items inclusive when non-empty
- end_of_conversation is a real boolean
"""

import pytest
from pydantic import ValidationError
from app.schemas import ChatResponse, ChatRequest, Message, Recommendation


class TestChatResponseSchema:
    """Verify ChatResponse always produces the correct schema."""

    def test_empty_recommendations(self):
        """recommendations should default to empty list, not null."""
        resp = ChatResponse(reply="Hello", end_of_conversation=False)
        data = resp.model_dump()
        assert data["recommendations"] == []
        assert data["recommendations"] is not None
        assert isinstance(data["end_of_conversation"], bool)

    def test_with_recommendations(self):
        """recommendations with valid items should serialize correctly."""
        resp = ChatResponse(
            reply="Here are my recommendations.",
            recommendations=[
                Recommendation(name="Test A", url="https://shl.com/a/", test_type="K"),
                Recommendation(name="Test B", url="https://shl.com/b/", test_type="P"),
            ],
            end_of_conversation=True,
        )
        data = resp.model_dump()
        assert len(data["recommendations"]) == 2
        assert data["end_of_conversation"] is True
        for rec in data["recommendations"]:
            assert "name" in rec
            assert "url" in rec
            assert "test_type" in rec

    def test_max_recommendations(self):
        """Should handle up to 10 recommendations."""
        recs = [
            Recommendation(name=f"Test {i}", url=f"https://shl.com/{i}/", test_type="K")
            for i in range(10)
        ]
        resp = ChatResponse(
            reply="Many recommendations.",
            recommendations=recs,
            end_of_conversation=False,
        )
        data = resp.model_dump()
        assert len(data["recommendations"]) == 10

    def test_end_of_conversation_is_bool(self):
        """end_of_conversation must be a real boolean, not a string or int."""
        resp = ChatResponse(reply="Done", recommendations=[], end_of_conversation=True)
        data = resp.model_dump()
        assert data["end_of_conversation"] is True
        assert isinstance(data["end_of_conversation"], bool)

        resp2 = ChatResponse(reply="More", recommendations=[], end_of_conversation=False)
        data2 = resp2.model_dump()
        assert data2["end_of_conversation"] is False
        assert isinstance(data2["end_of_conversation"], bool)

    def test_reply_is_string(self):
        """reply must be a string."""
        resp = ChatResponse(reply="Hello world", recommendations=[], end_of_conversation=False)
        data = resp.model_dump()
        assert isinstance(data["reply"], str)

    def test_all_required_fields_present(self):
        """All three top-level fields must be present in serialized output."""
        resp = ChatResponse(reply="Test", recommendations=[], end_of_conversation=False)
        data = resp.model_dump()
        assert set(data.keys()) == {"reply", "recommendations", "end_of_conversation"}

    def test_recommendation_fields(self):
        """Each recommendation must have exactly name, url, test_type."""
        rec = Recommendation(name="OPQ32r", url="https://shl.com/opq/", test_type="P")
        data = rec.model_dump()
        assert set(data.keys()) == {"name", "url", "test_type"}


class TestChatRequestSchema:
    """Verify ChatRequest handles various input shapes."""

    def test_valid_request(self):
        """Normal request with alternating user/assistant messages."""
        req = ChatRequest(messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
            Message(role="user", content="Help me"),
        ])
        assert len(req.messages) == 3

    def test_empty_messages(self):
        """Empty messages list should be valid (handled at app layer)."""
        req = ChatRequest(messages=[])
        assert len(req.messages) == 0

    def test_single_message(self):
        """Single user message should be valid."""
        req = ChatRequest(messages=[
            Message(role="user", content="I need assessments"),
        ])
        assert len(req.messages) == 1

    def test_all_assistant_messages(self):
        """All assistant messages (malformed) should still parse."""
        req = ChatRequest(messages=[
            Message(role="assistant", content="Hello"),
            Message(role="assistant", content="World"),
        ])
        assert len(req.messages) == 2

    def test_long_conversation(self):
        """Near the 8-message cap."""
        messages = []
        for i in range(7):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append(Message(role=role, content=f"Message {i}"))
        req = ChatRequest(messages=messages)
        assert len(req.messages) == 7
