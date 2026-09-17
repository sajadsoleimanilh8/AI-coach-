/**
 * Match selection: the picker, and ?match= surviving a refresh.
 *
 * Guards the failure this was written for -- landing on a match-scoped tab
 * with no match and finding a dead end, or selecting one and losing it on the
 * next reload. Needs the backend on :8000 and the dev server on :3000, and at
 * least one processed match in the database.
 */
import { chromium } from 'playwright';

const fail = [];
const ok = [];
const check = (cond, label) => (cond ? ok : fail).push(label);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));

// 1. Land straight on Simulation with no match, as the user did.
await page.goto('http://localhost:3000/#dashboard/simulation', { waitUntil: 'networkidle' });
await page.waitForTimeout(1200);

let body = await page.locator('body').innerText();
check(/Select a Match First/i.test(body), 'gate still appears when no match is selected');
const options = page.locator('.dash-match-option');
const count = await options.count();
console.log(`match options offered: ${count}`);
check(count > 0, 'picker lists real matches instead of a dead end');
check(/tracking rows/i.test(body), 'each row states whether it actually has tracking data');

// 2. Pick the first one that reports tracking rows.
const usable = page.locator('.dash-match-option', { has: page.locator('.dash-match-flag.is-good') }).first();
const chosenId = (await usable.locator('code').innerText()).trim();
await usable.click();
await page.waitForTimeout(2500);

body = await page.locator('body').innerText();
check(!/Select a Match First/i.test(body), 'gate clears after picking');
check(/Adjust a tactical parameter|Run Simulation|heuristic_proxy/i.test(body), 'Simulation content renders');

// 3. The selection must be in the URL.
const url = page.url();
console.log('url after pick:', url);
check(url.includes(`match=${chosenId}`), 'selection is written to ?match=');
check(url.includes('#dashboard/simulation'), 'hash route is untouched by the selection');

// 4. Refresh: the whole point.
await page.reload({ waitUntil: 'networkidle' });
await page.waitForTimeout(2500);
body = await page.locator('body').innerText();
check(!/Select a Match First/i.test(body), 'match survives a refresh');
check((await page.locator('.dash-tab.is-active .dash-tab-label').innerText()).toLowerCase().includes('simulation'),
      'still on Simulation after refresh');
// innerText reflects CSS text-transform, so compare case-insensitively.
const chipText = (await page.locator('.dash-match-chip code').first().innerText()).trim();
check(chipText.toLowerCase() === chosenId.toLowerCase(), 'header chip shows the restored match');

// 5. Back must return to the previous tab, NOT undo the selection.
await page.goto(`http://localhost:3000/?match=${chosenId}#dashboard/simulation`, { waitUntil: 'networkidle' });
await page.waitForTimeout(2000);
await page.locator('.dash-tab', { hasText: /Team Intelligence/i }).click();
await page.waitForTimeout(1200);
await page.goBack();
await page.waitForTimeout(1200);
check((await page.locator('.dash-tab.is-active .dash-tab-label').innerText()).toLowerCase().includes('simulation'),
      'back returns to Simulation');
body = await page.locator('body').innerText();
check(!/Select a Match First/i.test(body), 'back does not undo the match selection');

// 6. A dead id in the link must not masquerade as a selection.
await page.goto('http://localhost:3000/?match=does-not-exist#dashboard/simulation', { waitUntil: 'networkidle' });
await page.waitForTimeout(2500);
body = await page.locator('body').innerText();
check(/Select a Match First/i.test(body), 'unknown match id falls back to the picker');
check(!page.url().includes('does-not-exist'), 'dead id is cleared from the address bar');

// 7. The gate is shared, so every match-scoped tab must get the same picker,
//    and every one of them must accept a linked match.
for (const tab of ['player', 'team', 'calibration', 'simulation']) {
  await page.goto(`http://localhost:3000/#dashboard/${tab}`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1200);
  check(await page.locator('.dash-match-option').count() > 0, `${tab}: picker offers matches`);

  await page.goto(`http://localhost:3000/?match=${chosenId}#dashboard/${tab}`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(2500);
  const text = await page.locator('body').innerText();
  check(!/Select a Match First/i.test(text), `${tab}: opens directly on a linked match`);
  check(!/Restoring match/i.test(text), `${tab}: restore finishes rather than hanging`);
}

check(errors.length === 0, `no uncaught page errors (${errors.join(' | ') || 'none'})`);

console.log('\nPASS:');
ok.forEach((l) => console.log('   ok  ' + l));
if (fail.length) { console.log('\nFAIL:'); fail.forEach((l) => console.log('   XX  ' + l)); }
console.log(`\n${ok.length} passed, ${fail.length} failed`);
await browser.close();
process.exit(fail.length ? 1 : 0);
