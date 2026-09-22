# Lightspeed Stack UI

A [PatternFly Chatbot](https://www.patternfly.org/patternfly-ai/chatbot/overview) single-page chat interface for the Lightspeed Stack REST API. Built with Vite + React + TypeScript.

This UI talks to the backend exclusively over HTTP and holds no business logic of its own:

- **Models** come from `GET /v1/models`.
- **Chat** is a single session per page load, driven by `POST /v1/query` (streamed via SSE). The `conversation_id` returned by the `start` event is echoed back on the next turn so follow-ups stay attached to the same server-side agent run.
- There is intentionally **no conversation list/history sidebar** — the backend's `/v1/conversations` endpoints aren't implemented yet (see `lightspeed/app/endpoints/conversations.py`). Once that API exists, this UI can add a history nav backed by it.

## Develop

```sh
npm install
npm run dev
```

The dev server proxies `/v1/*` to `http://localhost:8080` (see `vite.config.ts`), so run the Lightspeed Stack backend (`lightspeed-stack.yaml` → `service.port`) alongside it.

## Build

```sh
npm run build
```

Outputs a static bundle to `dist/`. Serve it from the same origin as the API (or set `VITE_API_BASE_URL` at build time) so `/v1/*` requests resolve correctly.
