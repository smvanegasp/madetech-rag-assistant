# MadeTech RAG Assistant — Improvement Roadmap

> **Legend**
>
> - **[CLAUDE-CODE]** — Claude Code can do this (or heavily assist)
> - **[MANUAL]** — Requires manual/human effort (decisions, writing, external services)
> - **[COLLAB]** — Mixed: requires a human decision first, then Claude Code implements

---

## Phase 1 — Critical Fixes & Quick Wins

*Goal: Fix bugs, polish existing UX, and make small high-impact changes. These are low-risk, independent items that can be shipped and tested quickly.*

### 1.1 Frontend Bug Fixes — COMPLETED

- ~~**[CLAUDE-CODE]** Fix iPhone text box zoom bug (AWS only)~~ — Done: added `maximum-scale=1.0` to viewport meta + `text-base` (16px) on mobile input.
- ~~**[CLAUDE-CODE]** Fix iPhone text box growing too tall~~ — Done: textarea capped at 3 lines using cached line metrics, stable across all devices.
- ~~**[CLAUDE-CODE]** Fix iPad feedback button overlapping the message input~~ — Done: header button shown on all touch/tablet devices (`xl:hidden`), floating button desktop-only (`xl:flex`).
- ~~**[CLAUDE-CODE]** Fix AWS App Runner layout allowing minimal scroll~~ — Done: root uses `h-dvh`, html/body set to `overflow: hidden; height: 100dvh`.
- ~~**[COLLAB]** Verify tablet experience end-to-end~~ — Done: manually verified on iPhone and iPad.

**Additional improvements made during this session:**
- Textarea auto-grows naturally like WhatsApp (no letter clipping mid-line), with `overscroll-behavior: contain` to prevent scroll chaining on touch devices.
- Send button repositioned inline next to textarea (was in its own row below) for a cleaner, more space-efficient input bar.
- Welcome screen book icon moved inline next to "Knowledge Search" title to avoid clipping on short viewports (e.g., Chrome on iPhone).

### 1.2 Frontend Quick Enhancements — COMPLETED

- ~~**[CLAUDE-CODE]** Add favicon~~ — Done: inline SVG favicon (book icon on emerald-green rounded square) added to `index.html`.
- ~~**[COLLAB]** Give the assistant a name~~ — Done: named "Nexus". Updated page title, header fallback, welcome screen, welcome modal, and both backend system prompts (`TOOL_DECISION_SYSTEM_PROMPT`, `RAG_SYSTEM_PROMPT`).
- ~~**[CLAUDE-CODE]** Update first sample question~~ — Done: changed to "Tell me more about this app".
- ~~**[CLAUDE-CODE]** Add a "Try again" button on error~~ — Done: API errors show a retry button that re-sends the original user message with the error cleared from history.

### 1.3 System Prompt Improvements (Backend) — COMPLETED

- ~~**[CLAUDE-CODE]** Inject today's date dynamically~~ — Done: `{today}` placeholder in both `TOOL_DECISION_SYSTEM_PROMPT` and `RAG_SYSTEM_PROMPT`, formatted as "Month DD, YYYY" at runtime in `pipeline.py`.
- ~~**[COLLAB]** Add "Tell me about this app" capability~~ — Done: project repo link, creator info, and disclaimer already in prompt. Articles pending (see Pending Items).
- ~~**[COLLAB]** Add creator and disclaimer info~~ — Done: already present in original prompts (Sergio Vanegas, LinkedIn, academic project, disclaimer caveats).
- ~~**[CLAUDE-CODE]** Add repo link to the system prompt~~ — Done: `https://github.com/smvanegasp/madetech-rag-assistant` added to both prompts.
- ~~**[COLLAB]** Add a handbook summary to the system prompt~~ — Done: comprehensive summary covering all 150+ documents across 6 categories, including specific topics per area and an explicit list of topics the handbook does NOT cover. Also added the full handbook file index (organized by category and seniority level for roles) so Nexus can answer structural questions like "what roles exist?" or "what benefits are there?" directly from the system prompt without a RAG search.

### 1.4 Operational Quick Wins — COMPLETED

- ~~**[MANUAL]** Verify budget constraints on Groq and OpenAI APIs~~ — Done: Groq set to $5/month on-demand limit. OpenAI set to $5 pre-loaded credits with auto-recharge disabled.
- ~~**[MANUAL]** Monitor cost of serving the app~~ — Deferred: with $5/month caps on both providers, the budget limits themselves act as the guardrail. Revisit if caps are raised (see Pending Items).

---

## Phase 2 — Frontend UX & RAG Investigation

*Goal: Ship user-facing improvements and gather insights that will inform the SDK migration in Phase 3. These are independent of the backend architecture.*

### 2.1 Frontend UX Improvements — COMPLETED

- ~~**[COLLAB]** Evaluate and adjust welcome popup timing~~ — Done: 45-second auto-dismiss confirmed as appropriate.
- ~~**[COLLAB]** Add article/resource links in the UI~~ — Deferred: no articles available yet (see Pending Items).
- ~~**[CLAUDE-CODE]** Inline source citations (ChatGPT-style)~~ — Done: backend numbers chunks for LLM citation, `[n]` markers stripped from displayed text, clean "Sources:" list below each answer with sequential numbering and clickable doc links.
- ~~**[CLAUDE-CODE]** Add a Frequently Asked Questions section~~ — Done: collapsible "What can I help with?" section on welcome screen with 5 categories (Benefits, Roles, Ways of Working, Policies, Welfare) and a note on handbook limitations.
- ~~**[COLLAB]** Optional popup tutorial / onboarding GIFs~~ — Deferred: needs GIF content created first (see Pending Items).

**Additional improvements made during this session:**
- System prompt now uses human-readable names in the handbook file index ("Delivery Manager" not "delivery_manager") with explicit instruction to present names naturally.
- Nexus told it doesn't know anything about the user (no assumed name, role, or team).
- "Tell me about this app" now answered directly from system prompt — Nexus knows "this app" refers to itself.
- Structural questions (what roles exist, what benefits are there) answered from system prompt without triggering a RAG search.

### 2.2 Backend — RAG & Latency

- **[COLLAB]** Investigate double query rewriting: Check if the orchestrator LLM already rewrites the user query when calling the RAG tool, making the explicit `query_rewriting.py` step redundant. Log both the tool-call input and the rewritten query to compare. *(Claude Code can add logging; you analyze the results.)*
- **[COLLAB]** Test latency with alternative models: Benchmark response time and error rates with different LLMs (e.g., smaller Groq models, other providers via LiteLLM). Document results. *(Claude Code can build a benchmarking script; you run and evaluate.)*
- **[MANUAL]** Evaluate Supabase pause risk: Determine what happens if Supabase pauses your project after inactivity. Decide if chat logging is critical enough to warrant a paid tier or an alternative (e.g., simple file logging as fallback).

---

## Phase 3 — SDK Migration & Agent Architecture

*Goal: Migrate to a proper agent framework first, then build new tools on top of it. This avoids building tools twice — once in the current manual approach and again after migration.*

### 3.1 SDK Migration

- **[COLLAB]** Evaluate and migrate to OpenAI Agents SDK: Refactor the backend to use the OpenAI Agents SDK (or similar) for structured agent orchestration. This replaces the current manual tool-calling approach and gives better tool configuration, structured outputs, and multi-agent support. *(Major architectural decision — you plan the migration; Claude Code executes the refactor.)*
- **[CLAUDE-CODE]** Improve structured outputs with Pydantic: Once on the new SDK, define strict Pydantic models for all tool inputs/outputs to get reliable structured responses from the LLM.

### 3.2 New Backend Tools (post-migration)

*These were originally Phase 2.1. Moved here so they are built on the new SDK rather than the manual tool-calling approach.*

- **[CLAUDE-CODE]** Create a "send feedback" tool: Build a new tool that the LLM can call when the user wants to give feedback. It should collect the message and log it (via Supabase or Resend email). Wire it into the existing `contact_service.py`.
- **[CLAUDE-CODE]** Create a "get in touch" tool: Similar to above but for contact requests — the LLM detects intent, collects details, and triggers an email or logs the request.
- **[CLAUDE-CODE]** Add input guardrails: Implement validation so the assistant only answers questions in English that are related to MadeTech or the application. Reject or politely redirect off-topic or non-English queries.

### 3.3 Multi-Tool Orchestration

- **[CLAUDE-CODE]** Create a planning/execution tool: Build a tool that lets the orchestrator LLM plan multi-step queries. Example: *"Compare Lead Engineer vs Software Engineer"* → Plan: (1) RAG query for Lead Engineer, (2) RAG query for Software Engineer, (3) Compare and respond.
- **[CLAUDE-CODE]** Support multiple RAG tool calls: Enable the agent to make multiple independent RAG queries in a single user interaction (needed for comparison questions, multi-topic queries).
- **[COLLAB]** Handle diverse input types gracefully: Configure the orchestrator to recognize and route different input types appropriately:
  - Single question (single or multi-source)
  - Multiple independent questions
  - Unrelated / off-topic questions
  - Accidental typing / gibberish
  - General-purpose questions (answerable from system prompt)
  - Feedback or contact requests (route to tools from 3.2)
- **[COLLAB]** Experiment with tool configurations: Add/remove tools from the tool list and test behavior. Investigate whether the system prompt needs to explicitly describe available tools or if SDK-provided tool descriptions are sufficient.

### 3.4 Advanced Search

- **[COLLAB]** Add a web/SEO search tool: Create a tool that combines RAG results with live web search for more comprehensive answers. *(Decide on search provider — e.g., Serper, Tavily, Brave; Claude Code integrates it.)*

### 3.5 Tool-Calling Transparency (Frontend)

- **[CLAUDE-CODE]** Show the tool-calling pattern to the user: Display a collapsible "thinking" or "steps" section in the UI that shows which tools the LLM called and in what order (e.g., *"Searching handbook for Lead Engineer… Searching handbook for Software Engineer… Comparing results…"*). Requires the backend to stream or return tool-call metadata.

---

## Phase 4 — Content, Documentation & DevOps

*Goal: Improve everything around the code — articles, documentation, deployment pipeline. These are mostly human tasks that can happen in parallel with any phase.*

### 4.1 Content & Articles

- **[MANUAL]** Write articles that lead with the customer and the problem: In any blog posts or write-ups, start with the user pain point and your insight on why this matters, before describing the solution.
- **[MANUAL]** Create a one-pager: Write a concise document explaining what the app does, the problem it solves, and how to use it. Make the link available in the app.
- **[COLLAB]** Generate a comprehensive handbook summary: Create a PDF or document summarizing all handbook sections, role types, and key topics. Use an LLM to assist. This also feeds back into the system prompt (Phase 1.3).

### 4.2 Repository & Documentation

- **[CLAUDE-CODE]** Improve the README: Update `README.md` with clear setup instructions, architecture overview, screenshots, and links to the live app.
- **[CLAUDE-CODE]** Add repo link to the app UI: Make the GitHub link visible somewhere in the frontend.
- **[COLLAB]** Investigate CI/CD: Set up continuous integration and deployment (e.g., GitHub Actions → build → deploy to AWS App Runner or HuggingFace). *(You choose the pipeline; Claude Code writes the workflow files.)*

### 4.3 Infrastructure Considerations

- **[MANUAL]** Evaluate alternative document stores: Consider whether MongoDB or another vector DB option would be better than ChromaDB Cloud for your use case. Only worth exploring if you hit scaling or cost issues.
- **[MANUAL]** Evaluate whether the handbook source data needs updating: Check if the handbook markdown files in `backend/data/handbook/` are still current.
- **[MANUAL]** Decide on a web presence strategy: Consider building a dedicated page on your website for this project (keeping headers, linking front/back, and listing improvements).

---

## Summary View


| Phase | Focus                                         | Effort     | Items    |
| ----- | --------------------------------------------- | ---------- | -------- |
| **1** | Bug fixes, quick wins, system prompt          | Low–Medium | 14 items |
| **2** | Frontend UX, RAG investigation                | Medium     | 8 items  |
| **3** | SDK migration, new tools, multi-agent         | High       | 11 items |
| **4** | Content, docs, DevOps                         | Varies     | 8 items  |


---

## Suggested Workflow per Phase

1. **Plan:** Review items, make any human decisions needed (marked **[MANUAL]** or **[COLLAB]**).
2. **Implement:** Use Claude Code for all **[CLAUDE-CODE]** and **[COLLAB]** items.
3. **Test:** Manually verify on all target platforms (desktop, iPhone, iPad, HuggingFace, AWS).
4. **Ship:** Deploy and monitor before starting the next phase.
5. **Repeat.**

---

## Pending Items (Cross-Phase)

- **[MANUAL]** Write articles about the project: Blog posts, write-ups, or case studies to link from the app and system prompt. Once available, add URLs to the system prompt and UI. Also needed for article/resource links in the app UI. *(Referenced in Phase 1.3, Phase 2.1, and Phase 4.1.)*
- **[COLLAB]** Build popup tutorial / onboarding GIFs: Create a short walkthrough (2–4 steps) showing app features, then build a carousel component. *(Referenced in Phase 2.1.)*
- **[COLLAB]** Add cost monitoring: Log token counts per request to Supabase for usage tracking. Not needed while $5/month caps are in place — revisit if budget limits are raised. *(Referenced in Phase 1.4.)*