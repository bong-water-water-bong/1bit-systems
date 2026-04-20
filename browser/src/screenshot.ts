// screenshot.ts — open a fresh tab, load a URL, save a PNG, close the tab.
//
// Usage:  bun run src/screenshot.ts <url> [output.png]

import puppeteer from "puppeteer-core";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";

const PORT = process.env.HALO_BROWSER_PORT ?? "9222";
const URL = process.argv[2];
const OUT = process.argv[3] ?? "/tmp/halo-browser-shot.png";

if (!URL) {
  console.error("usage: bun run src/screenshot.ts <url> [output.png]");
  process.exit(1);
}

const browser = await puppeteer.connect({
  browserURL: `http://127.0.0.1:${PORT}`,
  defaultViewport: { width: 1400, height: 900 },
});

const page = await browser.newPage();
try {
  await page.goto(URL, { waitUntil: "networkidle2", timeout: 30_000 });
  mkdirSync(dirname(OUT), { recursive: true });
  await page.screenshot({ path: OUT, fullPage: true });
  console.log(`saved ${OUT}`);
} finally {
  await page.close();
  await browser.disconnect();
}
