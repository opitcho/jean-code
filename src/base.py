import ollama


class BaseAgent:
    """Turn-by-turn agent that chats with a user through an Ollama model.

    Each turn appends the user's message to the running transcript, sends the
    whole transcript to Ollama, appends the model's reply, and returns it.
    """

    def __init__(self, model: str, system_prompt: str = "", client: ollama.Client | None = None):
        self.model = model
        self.system_prompt = system_prompt
        self.client = client or ollama.Client()

        self.messages: list[dict] = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def step(self, user_input: str) -> str:
        """Send one user message, return the assistant's reply, updating history."""
        self.messages.append({"role": "user", "content": user_input})

        response = self.client.chat(model=self.model, messages=self.messages)
        message = response["message"]
        self.messages.append(message)

        return message["content"]

    def run(self):
        """Interactive REPL: read a line from the user, print the reply, repeat."""
        print(f"Chatting with {self.model}. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            reply = self.step(user_input)
            print(f"agent> {reply}")


if __name__ == "__main__":
    agent = BaseAgent(model="qwen3")
    agent.run()
