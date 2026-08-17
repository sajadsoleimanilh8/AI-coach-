                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

import { chromium } from 'playwright';

const BASE = process.env.BASE || 'http://localhost:3000';

const TABS = [
  { id: 'match', label: 'Match Analysis' },
  { id: 'player', label: 'Player Intelligence' },
  { id: 'team', label: 'Team Intelligence' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'simulation', label: 'Simulation' },
  { id: 'health', label: 'Pre-Match Health' },
  { id: 'psychology', label: 'Pre-Match Psychology' },
  { id: 'coach', label: 'Coach Chat' },
];

let passed = 0;
let failed = 0;

                                                                                                                                                                                                                
function labelMatches(rendered, expected) {
  return Boolean(rendered) && rendered.toLowerCase().includes(expected.toLowerCase());
}

function check(name, condition, detail = '') {
  if (condition) {
    passed += 1;
    console.log(`  PASS  ${name}`);
  } else {
    failed += 1;
    console.log(`  FAIL  ${name}${detail ? `  -- ${detail}` : ''}`);
  }
}

                                                                        
async function activeState(page) {
  return page.evaluate(() => {
    const active = document.querySelector('.dash-tab.is-active');
    const heading = document.querySelector('#dashboard .dash-section-head .section-headline');
    const tag = document.querySelector('#dashboard .dash-section-head .section-tag');
    return {
      activeLabel: active ? active.innerText.replace(/\s+/g, ' ').trim() : null,
      activeCount: document.querySelectorAll('.dash-tab.is-active').length,
      sectionTag: tag ? tag.innerText.trim() : null,
      headline: heading ? heading.innerText.replace(/\s+/g, ' ').trim() : null,
      view: document.body.dataset.view,
      hash: window.location.hash,
    };
  });
}

const run = async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(String(error)));

                                                                                    
  console.log('\n1. TAB HIT TARGETS (the reported bug)');
  await page.goto(`${BASE}/#dashboard`, { waitUntil: 'networkidle' });
  await page.waitForSelector('.dash-tab', { timeout: 10000 });

  const tabCount = await page.locator('.dash-tab').count();
  check(`all ${TABS.length} tabs rendered`, tabCount === TABS.length, `found ${tabCount}`);

                                                                            
                             
  const overlaps = await page.evaluate(() => {
    const bad = [];
    document.querySelectorAll('.dash-tab').forEach((tab) => {
      const label = tab.querySelector('.dash-tab-label, .dash-tab-short');
      if (!label) return;
      const box = label.getBoundingClientRect();
      const hit = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
      if (!tab.contains(hit)) {
        bad.push({ label: label.innerText.trim(), hit: hit ? hit.className : 'null' });
      }
    });
    return bad;
  });
  check('every label sits inside its own button', overlaps.length === 0, JSON.stringify(overlaps));

                                                                                 
  console.log('\n2. CLICKING EVERY TAB');
  for (const tab of TABS) {
    await page.locator(`.dash-tab:has-text("${tab.label}")`).first().click();
    await page.waitForFunction(
      (id) => window.location.hash === `#dashboard/${id}`,
      tab.id,
      { timeout: 4000 },
    ).catch(() => {});
    const state = await activeState(page);
    check(
      `click "${tab.label}" -> renders + URL`,
      state.hash === `#dashboard/${tab.id}` && state.activeCount === 1
        && labelMatches(state.activeLabel, tab.label),
      `hash=${state.hash} active=${state.activeLabel} headline=${state.headline}`,
    );
  }

                                                                                   
  console.log('\n3. DIRECT NAVIGATION (deep link to each tab)');
  for (const tab of TABS) {
    await page.goto(`${BASE}/#dashboard/${tab.id}`, { waitUntil: 'networkidle' });
    await page.waitForSelector('.dash-tab.is-active', { timeout: 8000 });
    const state = await activeState(page);
    check(
      `deep link #dashboard/${tab.id}`,
      state.view === 'dashboard' && labelMatches(state.activeLabel, tab.label),
      `view=${state.view} active=${state.activeLabel}`,
    );
  }

                                                                                
  console.log('\n4. REFRESH ON EACH TAB');
  for (const tab of TABS) {
    await page.goto(`${BASE}/#dashboard/${tab.id}`, { waitUntil: 'networkidle' });
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForSelector('.dash-tab.is-active', { timeout: 8000 });
    const state = await activeState(page);
    check(
      `reload on ${tab.id} stays put`,
      state.view === 'dashboard' && labelMatches(state.activeLabel, tab.label),
      `view=${state.view} active=${state.activeLabel}`,
    );
  }

                                                                                     
  console.log('\n5. BROWSER BACK / FORWARD');
  await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
  await page.locator('nav .nav-cta').click();                                      
  await page.locator('.dash-tab:has-text("Simulation")').first().click();
  await page.locator('.dash-tab:has-text("Coach Chat")').first().click();

  let state = await activeState(page);
  check('forward path lands on Coach Chat', state.hash === '#dashboard/coach', state.hash);

  await page.goBack({ waitUntil: 'load' });
  await page.waitForTimeout(400);
  state = await activeState(page);
  check('back -> Simulation', state.hash === '#dashboard/simulation'
    && labelMatches(state.activeLabel, 'Simulation'), `hash=${state.hash} active=${state.activeLabel}`);

  await page.goForward({ waitUntil: 'load' });
  await page.waitForTimeout(400);
  state = await activeState(page);
  check('forward -> Coach Chat', state.hash === '#dashboard/coach'
    && labelMatches(state.activeLabel, 'Coach Chat'), `hash=${state.hash} active=${state.activeLabel}`);

                                                    
  for (let i = 0; i < 4; i += 1) {
    await page.goBack({ waitUntil: 'load' }).catch(() => {});
    await page.waitForTimeout(200);
    state = await activeState(page);
    if (state.view === 'landing') break;
  }
  check('back reaches the landing view', state.view === 'landing', `view=${state.view}`);

                                                                             
  console.log('\n6. ROBUSTNESS');
  await page.goto(`${BASE}/#dashboard/not-a-real-tab`, { waitUntil: 'networkidle' });
  await page.waitForSelector('.dash-tab.is-active', { timeout: 8000 });
  state = await activeState(page);
  check(
    'unknown tab id falls back and corrects the URL',
    state.view === 'dashboard' && state.hash === '#dashboard/match',
    `hash=${state.hash} active=${state.activeLabel}`,
  );

  await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
  state = await activeState(page);
  check('bare / shows the landing view', state.view === 'landing', `view=${state.view}`);

  check('no uncaught page errors', pageErrors.length === 0, pageErrors.join(' | '));

  await browser.close();

  console.log(`\n${'='.repeat(56)}`);
  console.log(`  ${passed} passed, ${failed} failed`);
  console.log('='.repeat(56));
  process.exit(failed === 0 ? 0 : 1);
};

run().catch((error) => {
  console.error('test harness error:', error);
  process.exit(1);
});
