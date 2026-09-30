// Cloudflare Worker: live Kalshi prices for the Parlay Lab app.
//
// Kalshi's public API only answers browsers on kalshi.com, so the app can't ask it for
// prices directly. This worker forwards one read-only request, GET /markets?tickers=...,
// and adds the header that lets the app's page read the answer. It can't place orders or
// touch any account: it has no keys, and every other path is refused.
//
// Deploy: Cloudflare dashboard -> Workers & Pages -> Create -> Worker -> paste this file.

const KALSHI = "https://api.elections.kalshi.com/trade-api/v2/markets";
const ALLOWED_ORIGINS = [
  "https://fabemade.github.io",
  "http://localhost:8765",
];
const TICKER = /^[A-Z0-9.\-]+$/;

export default {
  async fetch(request) {
    const origin = request.headers.get("Origin") || "";
    const cors = {
      "Access-Control-Allow-Origin": ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0],
      "Access-Control-Allow-Methods": "GET, OPTIONS",
      "Vary": "Origin",
    };
    if (request.method === "OPTIONS") return new Response(null, { headers: cors });

    const url = new URL(request.url);
    if (request.method !== "GET" || url.pathname !== "/markets") {
      return new Response("Not found", { status: 404, headers: cors });
    }
    const tickers = (url.searchParams.get("tickers") || "").split(",").filter(Boolean);
    if (!tickers.length || tickers.length > 100 || !tickers.every(t => TICKER.test(t))) {
      return new Response("Pass 1-100 Kalshi tickers", { status: 400, headers: cors });
    }

    const upstream = await fetch(`${KALSHI}?limit=100&tickers=${tickers.join(",")}`, {
      headers: { "Accept": "application/json" },
      cf: { cacheTtl: 5 },   // brief edge cache: many phones asking at once = one Kalshi call
    });
    // Pass through only what the app needs.
    const data = await upstream.json().catch(() => ({ markets: [] }));
    const markets = (data.markets || []).map(m => ({
      ticker: m.ticker, status: m.status,
      yes_ask: m.yes_ask_dollars, no_ask: m.no_ask_dollars,
      yes_bid: m.yes_bid_dollars, no_bid: m.no_bid_dollars,
    }));
    return new Response(JSON.stringify({ markets, at: new Date().toISOString() }), {
      status: upstream.ok ? 200 : 502,
      headers: { ...cors, "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  },
};
