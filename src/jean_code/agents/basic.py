"""Two minimal root agents, for showing what a model does with and without tools.

`chat_agent` is the model alone: no tools, skills or subagents. `web_agent` adds `web_search` and `fetch_page`, and
nothing else. Every agent built here streams its replies.
"""

from jean_code.core.agent import DEFAULT_MODEL, Agent
from jean_code.tools.web import fetch_page, web_search

CHAT_PROMPT = """\
You are a helpful assistant. You have no tools: you can't search the web, open links or run code, so you answer \
from what you learned in training. If a question needs something you can't know, such as recent news or today's \
date, say so instead of guessing.
"""

WEB_PROMPT = """\
You are a helpful assistant that can search the web and read pages. For anything recent, specific or easy to get \
wrong (news, prices, versions, dates, figures), search with web_search, read the most relevant pages with \
fetch_page, then answer. A search snippet is a lead, not a source: read the page before you rely on it. Answer \
first, then list the URLs your facts came from, and say what you couldn't confirm. A question you can answer from \
general knowledge doesn't need a search.
"""


def chat_agent(model: str = DEFAULT_MODEL, **kwargs) -> Agent:
    """Returns a chat agent with no tools at all: the model answers from what it knows.

    Args:
        model: OpenRouter model id.
        **kwargs: Passed on to `Agent`, except `stream`.
    """
    return Agent(model, system_prompt=CHAT_PROMPT, stream=True, **kwargs)


def web_agent(model: str = DEFAULT_MODEL, **kwargs) -> Agent:
    """Returns a chat agent whose only tools are `web_search` and `fetch_page`.

    Neither tool needs approval.

    Args:
        model: OpenRouter model id.
        **kwargs: Passed on to `Agent`, except `stream`.
    """
    return Agent(
        model,
        system_prompt=WEB_PROMPT,
        tools={"web_search": web_search, "fetch_page": fetch_page},
        stream=True,
        **kwargs,
    )
