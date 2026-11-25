# Issue tracker

Track issues and specs as local Markdown under `.scratch/`.

- Use one directory per feature or workstream: `.scratch/<feature>/`.
- Store the spec at `.scratch/<feature>/spec.md`.
- Store each ticket in its own `issues/<NN>-<slug>.md` file, starting at `01`.
- Record a ticket's triage label in a `Status:` line near the top.
- Append discussion under `## Comments`.
- When publishing, create the corresponding file. When fetching, read the file identified by its path or ticket number.

## Wayfinder

- Store notes, decisions, and open questions in `.scratch/<effort>/map.md`.
- Store each child ticket in `issues/<NN>-<slug>.md`. Set `Type:` to `research`, `prototype`, `grilling`, or `task`.
- Record dependencies as `Blocked by: NN, NN`. A ticket becomes unblocked once all listed tickets are `resolved`.
- Take the lowest-numbered open, unblocked, unclaimed ticket. Set `Status: claimed` before starting.
- To resolve it, add `## Answer`, set `Status: resolved`, and link the decision from the map.
