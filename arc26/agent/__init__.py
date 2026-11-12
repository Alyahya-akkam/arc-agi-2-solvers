"""The agent solver (phase P6, ADR-020): a language model writes Python programs for a task.

The model writes ``transform(grid)`` in a per-task REPL; a sandboxed verifier re-runs each program
on every demonstration pair; only programs that reproduce every pair, and pass the hardcoding
guards, become attempts. The design is ``docs/research/agent_library_design.md``; the shared types
and interfaces are :mod:`arc26.agent.types`.
"""
