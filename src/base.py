import pydantic
from typing import Dict, List

DEFAULT_COMPACTION_THRESHOLD = 256_000
DEFAULT_COMPACTION_KEEP_RECENT_STEPS = 15


class BaseAgent:
    def __init__(self):


        self.messages = []
        self.queries = []
        self.client = None
        self.tools = None 


        # instructions, and the task statement that starts the run.
        self.system_prompt: str = ""
        self.task_prompt: str = ""


        while True:
            client.

    
    # TODO: Implement Skills
    self.skills = None




    def step(self):
        pass

    def maybe_compact_context(self) -> bool:
        pass


# TODO: Fetch from LLM inference engine
# TODO: First version should be able to at least do chat



