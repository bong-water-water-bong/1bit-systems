// attach.ts — connect to the halo-browser Chrome profile and report tabs.
//
// Prereq: launch-chrome.sh must be running (Chrome with --remote-debugging-port).
// Usage:  bun run src/attach.ts

import puppeteer from "puppeteer-core";

const PORT = process.env.HALO_BROWSER_PORT ?? "9222";
const URL = `http://127.0.0.1:${PORT}`;

async function main() {
  const browser = await puppeteer.connect({
    browserURL: URL,
    defaultViewport: null,
  });

  const pages = await browser.pages();
  console.log(`attached to ${URL} — ${pages.length} page(s) open`);
  for (const p of pages) {
    const t = await p.title().catch(() => "(no title)");
    console.log(`  ${p.url()}  —  ${t}`);
  }

  // IMPORTANT: disconnect, not close. close() would kill the user's Chrome.
  await browser.disconnect();
}

main().catch((e) => {
  console.error("halo-browser attach failed:", e.message);
  console.error(`is ${URL}/json/version reachable? run launch-chrome.sh first`);
  process.exit(1);
});
