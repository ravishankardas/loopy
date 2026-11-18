"""Test doubles shared by the test files."""

from loopy import actions


class FakeLlama:
    """Stands in for llama_cpp.Llama: replies with scripted text, streamed in
    small pieces like the real model, and records what it was sent."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []  # messages sent on each call

    def create_chat_completion(self, messages, stream=False, **_kwargs):
        self.calls.append(list(messages))
        reply = self.replies.pop(0) if self.replies else "[none] (no more scripted replies)"

        def chunks():
            for i in range(0, len(reply), 3):  # 3 characters at a time, like tokens
                yield {"choices": [{"delta": {"content": reply[i:i + 3]}}]}

        return chunks()


class RecordedActions:
    """Replaces actions.run so tests never touch the real Chrome."""

    def __init__(self):
        self.ran: list[str] = []

    def __call__(self, action: str) -> actions.Result:
        self.ran.append(action)
        return actions.Result(True, f"(pretend) did {action}")
