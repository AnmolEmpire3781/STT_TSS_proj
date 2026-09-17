A)A — Architecture inspection
Read AGENTS.md first.

Inspect the latest merged changes on main.

Do not modify application code.

Compare the current implementation against:

- AGENTS.md
- updates_sync_latest.md
- CHANGELOG_EN_HI.md where relevant

Identify:

- architecture changes
- changed API contracts
- changed provider interfaces
- changed frontend workflows
- new dependencies
- deprecated/fallback behavior
- instructions in AGENTS.md that are now stale

Important:
The current reviewed/merged main branch is the implementation
source of truth.

If merged code intentionally changes an older documented behavior,
do not propose reverting the code merely to satisfy stale documentation.

Propose the minimal documentation changes required.

Do not edit anything until I approve.
B)
Approved.

Update only:

- AGENTS.md

to reflect the current merged architecture.

Do not modify application code.

AGENTS.md must describe current architecture and active rules.

Historical/obsolete behavior should remain in the changelog rather
than being enforced as a current repository rule.

Keep useful comments and rationale.

Show me the diff when finished.

Do not commit.
Do not push.

c)
Then manually inspect:

git status

git diff -- AGENTS.md updates_sync_latest.md CHANGELOG_EN_HI.md

If correct:

git add AGENTS.md updates_sync_latest.md CHANGELOG_EN_HI.md

git commit -m "Sync repository instructions with current architecture"

git push origin main

Or do it through your normal PR process if you prefer all changes reviewed.
c) Starting new feature:

Starting a new feature

Never develop a significant feature directly on main.

Start:

git switch main

git pull --ff-only origin main

git switch -c feature/<feature-name>

Examples:

git switch -c feature/feedback-review-loop

or:

git switch -c feature/hinglish-auto-language

Read AGENTS.md first.

Inspect the current implementation relevant to this task.

Do not edit anything yet.

I want to implement:

<describe feature here>

Trace the existing end-to-end call path.

Identify:

- files that actually need modification
- API/schema changes
- frontend changes
- backend changes
- provider changes
- data/storage implications
- backward compatibility risks
- tests required

Preserve all current working functionality that is unrelated
to this feature.

Propose the smallest safe implementation plan.

Do not modify files yet.d changes
required while preserving current working functionality.

D)Once its plan makes sense:

Approved. Implement the proposed changes.

You may edit the necessary repository files and run non-destructive
validation.

Do not commit.
Do not push.
Do not make unrelated refactors.

After finishing:

- show files changed
- summarize behavior changes
- run backend checks
- run frontend build if applicable
- report anything not tested

Then review:

git status
git diff --stat
git diff
Only after you're satisfied:

git add .
git commit -m "Your meaningful commit message"
git push -u origin <your-feature-branch>
