# frontend/

Browser UI for the web app layer. No build step — plain HTML, CSS, and JS loaded directly by FastAPI's static file mount.

## Files

| File | Role |
|---|---|
| `index.html` | Single-page shell; defines two UI panels — one that calls the API, one with a client-side fallback |
| `script.js` | Two independent sections: (1) async fetch to `POST /api/predict`, (2) keyword-based classifier that runs in the browser with no network call |
| `styles.css` | Two CSS blocks concatenated: a minimal dark theme (used by the API panel) and a styled theme (used by the standalone panel) |

## Key design decision

`script.js` contains two separate listeners that bind to different element IDs. The first requires the FastAPI server to be running; the second works offline. They are independent — neither calls the other.

## To change the UI

Any edits to `index.html` or `script.js` are reflected immediately on page reload (no build needed). The FastAPI server must be running for the `/api/predict` calls to work.
