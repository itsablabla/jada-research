# AGENTS.md — universal operating principles (Garza OS)

You are an agent operating inside an `itsablabla` project. These principles
apply to **every** task, before and above any repo-specific instructions.

## 1. Skills-first

List `.agents/skills/` (or the equivalent skill directory for the current
runtime) before acting. Match the task to the narrowest relevant skill and
follow it as a checklist. Never freelance if a skill exists. The canonical
entry point is the `default-workflow` skill.

## 2. Plan-first

Produce a written plan (todo list or short markdown) before any
state-changing tool call. The plan must include:

- **Goal** — one sentence.
- **Ordered steps** — with the skill or tool per step.
- **Test plan** — concrete commands / assertions, not "looks good".
- **Debug fallback** — ordered layers to check when the test fails.

## 3. Composite tools before hand-rolled code

This org has a rich MCP inventory (Composio for GitHub/SaaS, Firecrawl +
Tavily for web, Hyperbrowser for browser automation, Vault for secrets,
NextCloud for files/calendar/contacts, Context7 for package docs,
Hostinger for VPS, etc.). Before shelling out or writing a new HTTP
client, check the `composite-tools` skill and list available MCP
servers. Prefer an existing composite tool over hand-rolled code.

## 4. Fabric AI is the memory

Read Fabric (https://api.fabric.so) at the start of every task whose
subject might have prior context. Write to Fabric at the end of every
task that produced new knowledge, decisions, or infra changes. See the
`fabric-memory` skill for recipes and the handoff folder ID.

## 5. Test + debug plan is part of the task

Every dev task must include concrete test commands (not vibes) and an
ordered debug fallback (which layer to check first, second, third)
before coding starts. Don't claim "done" without executing the test
plan and recording the result.

## 6. Delegate substantive code to Claude Code

For multi-file edits, new features, refactors, or anything beyond a
single-file tweak, use the `dev-with-claude-code` skill instead of
writing code inline. Always review the diff and rerun the test plan
after Claude Code produces changes.

## Completion contract

A task is only "done" when **all** of these are true:

- [ ] Plan was written before execution.
- [ ] Matched skill was followed end-to-end.
- [ ] Composite tools were used where they existed.
- [ ] Test plan was executed and the result recorded.
- [ ] Fabric AI memory was read at start, written at end.
- [ ] If code changed, Claude Code produced the diff unless the change
      was a genuinely single-file tweak.

## Do NOT

- Commit secrets, API keys, or tokens to this repo.
- Freelance a task when a matching skill exists.
- Skip the plan-first / test-plan / Fabric-write steps because the
  task "seems small".
- Hand-roll HTTP clients for SaaS APIs that already have a composite
  tool / MCP server available.
