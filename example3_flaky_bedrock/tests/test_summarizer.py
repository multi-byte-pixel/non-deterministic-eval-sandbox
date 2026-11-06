from app.bedrock_client import get_client
from app.summarizer import summarize_ticket

TICKET = (
    "Hi, I reset my password this morning and now I can't log in at all. "
    "It says my account is locked. Can you unlock it? Thanks, Dana"
)


def test_summarize_login_ticket():
    summary = summarize_ticket(TICKET, get_client())
    assert summary == "Customer reports login failure after password reset; requests account unlock."
