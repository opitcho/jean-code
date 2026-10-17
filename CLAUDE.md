# jean-code

## Docstrings

Docstrings follow the [Google Python Style Guide §3.8](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings),
with the rules below on top.

### What a docstring says
1. What the object is, and what it promises its caller: enough to use it without reading its code. Side effects,
   errors, blocking and thread safety are promises; say them.
2. Only what the object itself contains or does. Who calls it, who listens to it, how its result travels and what
   another module does with it are documented where that happens.
3. No private mechanism: no `_private` names, locks, threads or algorithm steps, unless a caller must know. Then say
   it as a promise ("Thread-safe.", "Waits until a client answers.").
4. Each fact once. A module docstring doesn't repeat its class's, and a class doesn't repeat its methods'.
5. Refer to other code by name (`AgentNode`), not by file (`node.py`). Names can be searched; files move.

A maintainer's invariant ("this is safe because…") is a one-line comment at the line it protects, not docstring
text.

### Shape
- **Summary**: one line of at most 80 characters, ending in a period. A longer docstring continues after a blank
  line, wrapped at 120, with the closing quotes on their own line.
- **Mood**: descriptive, in every file: "Fetches rows from…", "Starts a turn on `text`.", "Returns the result…".
  Not imperative ("Fetch…"), and not a bare noun phrase ("The rows…"), except on properties.
- **Module**: the summary, what the module exports, and optionally a short usage example.
- **Class**: the summary says what an instance represents ("The address of a cheese shop.", never "Class
  that…"). An exception says what it represents ("No more cheese is available."), not "Raised when…".
  Constructor arguments go in `Args:`. Every public attribute goes in `Attributes:`, never in an end-of-line
  comment; `model_config` and discriminators (an event's `type`, a message's `role`) are left out. An argument
  kept unchanged as the attribute of the same name is documented once, in `Args:`, and the body says "The
  arguments it keeps are attributes of the same name."
- **Function or method**: every public one outside `ui/` has
  - `Args:`, with every parameter;
  - `Returns:` (or `Yields:`), unless it returns None, or the summary starts with "Returns" and says it all;
  - `Raises:`, with the exceptions a caller would handle.

  In `ui/` these sections are optional, since nothing else consumes the UI's functions. A private helper
  (`_name`) may go without a docstring; if it has one, the content and shape rules still apply.
- **Property**: reads like an attribute ("The node's place in the tree."), not "Returns…".
- **Override or dunder**: no docstring unless it changes the contract it inherits.
- **Tests**: a docstring only when it adds something the test's name doesn't say.
- **`@tool` functions**: their docstrings are prompts. The model sees them as the tool's schema, so they are
  written for the model and changed only as a prompt change.

Example:

```python
def cancel(self, path: str = "main") -> None:
    """Stops the agent at `path`, and every agent below it, after its current call.

    The main agent's job ends "interrupted"; any other agent stays cancelled until it's resumed.

    Args:
        path: The agent's path, e.g. "main/1".

    Raises:
        UnknownAgent: No agent in the tree has `path`.
    """
```
