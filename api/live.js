// Server-side proxy for ESPN's public soccer scoreboard API.
//
// A real browser can't call ESPN's endpoint directly: its WAF 403s any
// request carrying a normal browser User-Agent (or no recognized one at
// all), but passes a handful of known non-browser tool signatures - the
// same reason scripts/fetch_live_scores.py leaves Python urllib's default
// UA alone instead of overriding it. This function makes the same kind of
// request server-side (Vercel's Node runtime, not a browser), so the
// client-side "live score" poller in index.html can call this same-origin
// path every minute without hitting that block.
//
// Uses the built-in https module (not global fetch) so this doesn't depend
// on which Node major version the project happens to run on.
const https = require("https");

const ESPN_SLUGS = new Set([
  "eng.1", "eng.2", "esp.1", "ger.1", "ita.1", "fra.1", "ned.1", "tur.1", "por.1",
]);

function getJson(url) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { "User-Agent": "curl/8.4.0" } }, (r) => {
      let body = "";
      r.on("data", (c) => (body += c));
      r.on("end", () => {
        if (r.statusCode < 200 || r.statusCode >= 300) {
          reject(new Error(`espn ${r.statusCode}`));
          return;
        }
        try {
          resolve(JSON.parse(body));
        } catch (e) {
          reject(e);
        }
      });
    });
    req.on("error", reject);
    req.setTimeout(15000, () => req.destroy(new Error("espn timeout")));
  });
}

module.exports = async function handler(req, res) {
  const { slug, dates } = req.query;
  if (!ESPN_SLUGS.has(slug) || !/^\d{8}$/.test(dates || "")) {
    res.status(400).json({ error: "bad slug/dates" });
    return;
  }
  try {
    const url = `https://site.api.espn.com/apis/site/v2/sports/soccer/${slug}/scoreboard?dates=${dates}`;
    const data = await getJson(url);
    res.setHeader("Cache-Control", "s-maxage=20, stale-while-revalidate=40");
    res.status(200).json(data);
  } catch (e) {
    res.status(502).json({ error: String(e) });
  }
};
