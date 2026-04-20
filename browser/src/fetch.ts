// fetch.ts — borrow the attached session's cookies to fetch an authenticated URL.
//
// Unlike bare `fetch()`, this runs inside the Chrome context so Reddit /
// Discord / GH session cookies are in play. Dumps response body to stdout.
//
// Usage:  bun run src/fetch.ts <url>

import puppeteer from "puppeteer-core";

const PORT = process.env.HALO_BROWSER_PORT ?? "9222";
const URL = process.argv[2];

if (!URL) {
  console.error("usage: bun run src/fetch.ts <url>");
  process.exit(1);
}

const browser = await puppeteer.connect({
  browserURL: `http://127.0.0.1:${PORT}`,
  defaultViewport: null,
});

const page = await browser.newPage();
try {
  const res = await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 30_000 });
  const status = res?.status() ?? 0;
  const body = await page.evaluate(() => document.documentElement.outerHTML);
  console.error(`[halo-browser] ${status} ${URL} (${body.length} bytes)`);
  process.stdout.write(body);
} finally {
  await page.close();
  await browser.disconnect();
}
