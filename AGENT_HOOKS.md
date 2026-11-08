# Agent memory hooks

Claude Code, Codex CLI, and Nous Research Hermes Agent can recall prior work
from this engine before a turn and save the prompt plus final response when the
turn completes. All three share the same project/branch memory, so work captured
in one agent can be recalled in another. A generic JSON adapter supports other
runtimes that can invoke a command before and after a turn.

This is session context, not fact extraction or proof of successful execution.
The bridge does not read transcripts, tool outputs, files, or credentials from
`.env`. It stores at most 12,000 characters each of the prompt and final response.
Retrieval uses English PostgreSQL full-text search (stemming and query terms),
not semantic embeddings; paraphrases without shared terms may not match. Recall
returns at most three excerpts and injects at most 6,000 characters, labeled as
unverified historical data. No extra LLM or embedding API call is required.

## 1. Start the updated engine and grant access

Rebuild/restart the memory engine using the existing deployment instructions.
Its normal startup migration runner applies `003_agent_hooks.sql` to both new
and existing databases, along with all later repository migrations. The current
extension migration `009` requires role-creating database privileges even when
only hooks are used; follow [startup and upgrade instructions](REMEDIATION.md#start-or-upgrade).
Existing memory tables are preserved.

Create a dedicated random bearer token (at least 32 characters). Add its grant
to the existing `MEMORY_AUTH_TOKENS` JSON mapping in `.env.auth`, preserving the
other tokens, then recreate the engine so it loads the updated environment:

```json
{
  "REPLACE_WITH_RANDOM_TOKEN": {
    "permissions": ["hooks:read", "hooks:write"],
    "projects": {"my-project": ["main"]}
  }
}
```

The token grants access to session excerpts shared by all agents in those exact
project/branch scopes. Add branches explicitly as needed. A read-only token can
use `hooks:read` alone, but capture will fail with an HTTP error. Do not give the
bridge the admin token. Each agent process must inherit the token environment:

```bash
export MEMORY_HOOK_TOKEN='your-dedicated-token'
```

## 2. Create a bridge config outside version control

Use Python 3.10 or later; the bridge uses only the standard library. Save this
as an absolute path such as `~/.config/pg-agent-memory/my-project.json`:

```json
{
  "enabled": true,
  "engine_url": "http://127.0.0.1:8001",
  "token_env": "MEMORY_HOOK_TOKEN",
  "project_id": "my-project",
  "workspace_root": "/absolute/path/to/my-project",
  "state_dir": "~/.local/state/pg-agent-memory-hooks"
}
```

`workspace_root` must be the Git checkout root. The bridge checks the event's
working directory and Git root, then reads the current branch. Other checkouts,
nested repositories, and detached HEADs are skipped. Configure each worktree
separately. This guard also prevents a global agent hook from capturing unrelated
projects. The server independently checks the token's project and branch grants.
Remote engines require HTTPS; HTTP is permitted only for loopback. HTTP redirects
are refused so bearer credentials are not forwarded to another endpoint.

## 3. Generate and merge agent configuration

From this repository, run the appropriate command below. The command prints a
snippet with absolute paths to this script, your Python interpreter, and the
bridge config. It does not edit agent settings or overwrite existing hooks.

```bash
python3 tools/agent_hooks.py config --agent claude --config ~/.config/pg-agent-memory/my-project.json
python3 tools/agent_hooks.py config --agent codex --config ~/.config/pg-agent-memory/my-project.json
python3 tools/agent_hooks.py config --agent hermes --config ~/.config/pg-agent-memory/my-project.json
```

| Agent | Merge generated entries into | Recall event | Capture event |
|---|---|---|---|
| Claude Code | `.claude/settings.local.json` or `~/.claude/settings.json` | `UserPromptSubmit` | `Stop` |
| Codex CLI | `.codex/hooks.json` or `~/.codex/hooks.json` | `UserPromptSubmit` | `Stop` |
| Hermes | Profile `config.yaml`, under `hooks:` | `pre_llm_call` | `post_llm_call` |

Append to the existing event arrays; preserve Entire and other hook entries.
For Claude/Codex, keep the existing top-level settings and merge the `hooks`
object rather than adding a second one. For Hermes, append YAML list entries
under the existing event names rather than adding another `hooks:` key.
Restart/reload the agent and use its normal hook review/consent flow. Do not
enable automatic approval just to install this bridge. Regenerate snippets if
you move this checkout or Python interpreter.

The configurations use a 10-second host timeout; HTTP calls time out after
3 seconds. Commands return `{}` with exit code zero on errors, and a short error
class on stderr without event content. They never block a prompt, approve a tool,
or request that the agent continue. Capture is synchronous and best effort:
there is no background delivery queue. An outage may lose that completion unless
the host redelivers it. A successful recall is not evidence that capture worked.

## Capture identity and local state

Codex and Hermes correlate turns using their native `turn_id`. Claude's
documented payload has no turn ID, so the bridge creates one at each prompt and
pairs Stop with the latest prompt for that session. This assumes ordered,
non-overlapping turns within a Claude session; duplicate prompt deliveries and
late Stop events after a new prompt cannot be distinguished reliably. Different
sessions are isolated. Stop events with `stop_hook_active` are skipped to avoid
recording continuation-loop output. Subagents, interrupts, and failed turns are
not captured by these configurations.

Only string prompts/responses are supported. Multimodal content is skipped.
The local SQLite database retains prompt correlation state for up to 24 hours
(expired rows are removed on the next hook invocation). Its directory is mode
0700 and the database is mode 0600. A prompt is saved before recall so a failed
recall does not prevent a later capture. Repeated completion delivery with the
same identity and content is a no-op in PostgreSQL, including concurrent retries.
Different content with the same identity returns 409 rather than overwriting it.

## Verify and remove memory

Start a turn about a distinctive subject and let the agent finish normally.
Then search the same project and branch through `/docs` or the API:

```bash
curl --fail http://127.0.0.1:8001/hooks/memories/search \
  -H "Authorization: Bearer $MEMORY_HOOK_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"my-project","git_branch":"main","query":"distinctive subject"}'
```

Results include `memory_id`, source agent, session/turn IDs, capture time, and
`verified: false`. A subsequent matching prompt should receive that historical
context. Check stderr if a hook is skipped; common causes are a missing token,
an unapproved host hook, a branch outside the token grant, or an unavailable API.

Stored excerpts persist until explicitly deleted. Prompts and final responses
can contain sensitive information: enable the hook only in intended projects.
To remove a session, first disable its bridge config (`"enabled": false`) or
remove its host entries and let in-flight hooks finish. Then POST to
`/hooks/memories/forget` with `project_id`, `git_branch`, `source`, and `session_id`.
This requires `hooks:write`. It deletes only that exact session scope. Clear the
corresponding local state directory when retiring the integration. Forget is not
a durable tombstone: re-enabling capture or replaying an event can store it again.

Hook excerpts are scoped by project/branch/source/session, independently of
companion users. Companion forgetting does not delete these excerpts, extension
evidence/checkpoints, or workspace files. Deleting a hook session likewise does
not remove records from those other domains or external exports/backups.

## Other agents

Invoke `python3 /absolute/path/tools/agent_hooks.py run --agent generic --config
/absolute/path/config.json` with one JSON object on stdin:

```json
{"hook_event_name":"prompt","cwd":"/absolute/path/to/my-project","session_id":"s1","turn_id":"t1","prompt":"Investigate lease expiry"}
```

The result is `{"context":"..."}` or `{}`. Your runtime must inject the context.
After completion, invoke it again with the same IDs and event `completed`, plus
`last_assistant_message` containing the final response. This adapter is an
extension contract, not a claim of native support for every other agent.

## Validation and upstream contracts

The deterministic suite exercises protocol fixtures for all four adapters,
scope rejection, bounded output, failure behavior, native turn correlation,
and real PostgreSQL storage, concurrency, authorization, retrieval, and deletion:

```bash
.venv-tests/bin/python tests/run_isolated.py --engine-python .venv-engine/bin/python
```

Host configuration and wire formats were checked against the official
[Claude Code hooks reference](https://code.claude.com/docs/en/hooks),
[Codex hooks reference](https://learn.chatgpt.com/docs/hooks), and
[Hermes shell-hook reference](https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks)
on 2026-10-20. The tests simulate documented hook events; they do not establish
live compatibility with every installed agent version. Use versions exposing
these events and fields, and smoke-test each host after installation.

The October 24 [final review](DESIGN_REVIEW.md) reran the complete deterministic
suite, including these hook fixtures, and clarified the shared migration and
deletion boundaries. It did not reinstall hooks or run live host compatibility
checks. Captured responses remain unverified history and are not automatically
promoted into the extension service's verified procedure recall.
