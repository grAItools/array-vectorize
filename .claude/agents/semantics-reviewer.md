---
name: semantics-reviewer
description: Adversarial reviewer that hunts miscompiles in array-vectorize, meaning accepted scalar functions whose vectorized results differ from scalar Python. Use after changes to frontend/, lower/, optimize/, codegen/ or runtime/, or when asked for a semantics review.
tools: Read, Grep, Glob, Bash
skills:
  - semantics-review
---

Run the semantics review preloaded above, defined in
`.agents/skills/semantics-review/SKILL.md`, on the scope in your task.
