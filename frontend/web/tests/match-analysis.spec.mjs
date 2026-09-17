/**
 * The Match Analysis tab, against a REAL processed match, in a real browser.
 *
 *   MATCH=<match_id> BASE=http://localhost:4173 node tests/match-analysis.spec.mjs
 *
 * Covers the five things that were reported broken and that a unit test
 * cannot honestly claim to verify, because each one is a property of the
 * rendered page and its network traffic rather than of a function:
 *
 *   1. the player loads the PROCESSED video, not the raw upload
 *   2. that video actually decodes in the browser (readyState + real
 *      dimensions + duration, not merely a <video src> that 404s quietly)
 *   3. the tactical overlay is inside the same stacking container as the
 *      video and shares its box -- the "overlay is on the wrong video"
 *      failure was invisible to every test that only checked it existed
 *   4. the overlay's coordinate system is the SOURCE frame, so markers land
 *      inside the picture instead of off its right-hand edge
 *   5. the events timeline renders and seeks the video when clicked
 *
 * Needs a backend with at least one completed match. Exits 2 (skipped, not
 * failed) when there is none, so it cannot pass vacuously.
 */

import { chromium } from 'playwright';

const BASE = process.env.BASE || 'http://localhost:3000';
const API = process.env.API || 'http://localhost:8000';

let passed = 0;
let failed = 0;

function check(name, condition, detail = '') {
  if (condition) {
    passed += 1;
    console.log(`   ok  ${name}`);
  } else {
    failed += 1;
    console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

/** A match that has actually been through the pipeline AND has a render. */
async function pickMatch() {
  if (process.env.MATCH) return process.env.MATCH;
  const list = await (await fetch(`${API}/api/matches`)).json();
  for (const item of list) {
    if (item.job_status !== 'completed' || !item.tracking_rows) continue;
    const summary = await (await fetch(`${API}/api/matches/${item.match_id}`)).json();
    if (summary.processed_video_exists) return item.match_id;
  }
  return null;
}

const matchId = await pickMatch();
if (!matchId) {
  console.log('SKIP: no completed match with a processed render on this backend.');
  process.exit(2);
}
console.log(`match under test: ${matchId}\n`);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });

const pageErrors = [];
page.on('pageerror', (error) => pageErrors.push(String(error)));

// Every video request the page makes, so "which clip is it playing" is
// answered by observed traffic rather than by reading the DOM's src.
const videoRequests = [];
page.on('request', (request) => {
  const url = request.url();
  if (url.includes('/api/videos/')) videoRequests.push(url);
});

await page.goto(`${BASE}/?match=${matchId}#dashboard/match`, { waitUntil: 'networkidle' });
await page.waitForTimeout(2500);

/* -- 1. the processed clip is what the player chose ------------------- */
console.log('1. VIDEO SOURCE');
const src = await page.getAttribute('video.dash-video', 'src');
check('a video element is rendered', Boolean(src), String(src));
check('src points at the PROCESSED render, not /file',
  Boolean(src && src.includes('/processed')), String(src));
check('the processed URL was actually requested',
  videoRequests.some((url) => url.includes('/processed')),
  videoRequests.join(' | '));

/* -- 2. it really decodes ---------------------------------------------- */
console.log('\n2. THE BROWSER CAN DECODE IT');
const media = await page.evaluate(async () => {
  const video = document.querySelector('video.dash-video');
  if (!video) return null;
  if (video.readyState < 1) {
    await new Promise((resolve) => {
      video.addEventListener('loadedmetadata', resolve, { once: true });
      setTimeout(resolve, 8000);
    });
  }
  return {
    readyState: video.readyState,
    width: video.videoWidth,
    height: video.videoHeight,
    duration: video.duration,
    errorCode: video.error ? video.error.code : null,
  };
});
check('no MediaError on the element', media && media.errorCode === null,
  media ? `error code ${media.errorCode}` : 'no element');
check('metadata loaded (readyState >= 1)', Boolean(media && media.readyState >= 1),
  media ? `readyState ${media.readyState}` : '');
check('real decoded dimensions', Boolean(media && media.width > 0 && media.height > 0),
  media ? `${media.width}x${media.height}` : '');
check('finite duration', Boolean(media && Number.isFinite(media.duration) && media.duration > 0),
  media ? `${media.duration}` : '');

/* -- 3. the overlay is attached to THAT video -------------------------- */
console.log('\n3. OVERLAY IS ON THE PLAYED VIDEO');
const geometry = await page.evaluate(() => {
  const video = document.querySelector('video.dash-video');
  const overlay = document.querySelector('svg.dash-video-overlay');
  if (!video || !overlay) return { overlay: Boolean(overlay) };
  const v = video.getBoundingClientRect();
  const o = overlay.getBoundingClientRect();
  return {
    overlay: true,
    // Same stacking container: the overlay's parent must contain the video.
    sameContainer: overlay.parentElement === video.parentElement,
    dx: Math.abs(v.x - o.x),
    dy: Math.abs(v.y - o.y),
    dw: Math.abs(v.width - o.width),
    dh: Math.abs(v.height - o.height),
    viewBox: overlay.getAttribute('viewBox'),
  };
});
check('the overlay is rendered at all', geometry.overlay === true);
check('overlay and video share a stacking container', geometry.sameContainer === true);
check('overlay box matches the video box',
  geometry.dx <= 2 && geometry.dy <= 2 && geometry.dw <= 2 && geometry.dh <= 2,
  `dx=${geometry.dx} dy=${geometry.dy} dw=${geometry.dw} dh=${geometry.dh}`);

/* -- 4. coordinates land inside the picture ---------------------------- */
console.log('\n4. OVERLAY COORDINATE SPACE');
const summary = await (await fetch(`${API}/api/matches/${matchId}`)).json();
const [vbW, vbH] = (geometry.viewBox || '0 0 0 0').split(' ').slice(2).map(Number);
check('viewBox is the SOURCE frame extent, not the played video size',
  vbW === summary.frame_width_px && vbH === summary.frame_height_px,
  `viewBox ${vbW}x${vbH} vs source ${summary.frame_width_px}x${summary.frame_height_px} `
  + `(played ${media && media.width}x${media && media.height})`);

const markers = await page.evaluate(() => {
  const overlay = document.querySelector('svg.dash-video-overlay');
  if (!overlay) return null;
  const box = overlay.getBoundingClientRect();
  const circles = [...overlay.querySelectorAll('circle')];
  const outside = circles.filter((circle) => {
    const r = circle.getBoundingClientRect();
    // Centre of each marker, against the overlay's own painted box.
    const cx = r.x + r.width / 2;
    const cy = r.y + r.height / 2;
    return cx < box.x - 1 || cx > box.x + box.width + 1
        || cy < box.y - 1 || cy > box.y + box.height + 1;
  });
  return { total: circles.length, outside: outside.length };
});
check('overlay drew markers', Boolean(markers && markers.total > 0),
  markers ? `${markers.total} markers` : 'none');
check('no marker falls outside the video box',
  Boolean(markers && markers.outside === 0),
  markers ? `${markers.outside} of ${markers.total} outside` : '');

/* -- 5. events timeline, and it seeks ---------------------------------- */
console.log('\n5. EVENTS TIMELINE');
const eventsApi = await (await fetch(`${API}/api/matches/${matchId}/events`)).json();
const rows = await page.$$('.dash-event-row');
if (eventsApi.total === 0) {
  check('no events in the API and none rendered (honest empty state)', rows.length === 0);
} else {
  check('event rows are rendered', rows.length > 0, `${rows.length} rows for ${eventsApi.total} events`);

  const before = await page.evaluate(() => document.querySelector('video.dash-video').currentTime);
  await rows[Math.min(1, rows.length - 1)].click();
  await page.waitForTimeout(900);
  const after = await page.evaluate(() => document.querySelector('video.dash-video').currentTime);
  check('clicking an event seeks the video', Math.abs(after - before) > 0.05,
    `currentTime ${before} -> ${after}`);
}

/* -- 6. heatmap renders without being asked ---------------------------- */
console.log('\n6. HEATMAP');
const heat = await page.evaluate(() => {
  const cells = document.querySelectorAll('.dash-pitch rect');
  const empty = [...document.querySelectorAll('.dash-empty-title, h4')]
    .some((node) => /No Player Selected/i.test(node.textContent || ''));
  return { cells: cells.length, awaitingSelection: empty };
});
check('the heatmap does not sit waiting for a manual player id',
  heat.awaitingSelection === false);
check('heatmap drew cells', heat.cells > 0, `${heat.cells} rects`);

/* -- 7. the overlay actually follows the video ------------------------- */
console.log('\n7. OVERLAY SYNCHRONISATION');

/** The frame number the overlay is currently showing, from the readout. */
async function shownFrame() {
  const text = await page.textContent('.dash-frame-readout');
  const match = /frame\s*(\d+)/i.exec(text || '');
  return match ? Number(match[1]) : null;
}

await page.evaluate(() => {
  const video = document.querySelector('video.dash-video');
  video.currentTime = 0;
});
await page.waitForTimeout(600);
const atZero = await shownFrame();

// PLAY. The overlay must advance on its own, driven by the video clock.
await page.evaluate(() => document.querySelector('video.dash-video').play());
await page.waitForTimeout(2000);
const whilePlaying = await shownFrame();
await page.evaluate(() => document.querySelector('video.dash-video').pause());
check('overlay advances while the video plays',
  atZero !== null && whilePlaying !== null && whilePlaying > atZero,
  `frame ${atZero} -> ${whilePlaying}`);

// SEEK BACKWARDS while paused. `timeupdate` alone would not repaint here.
await page.evaluate(() => { document.querySelector('video.dash-video').currentTime = 0.5; });
await page.waitForTimeout(700);
const afterRewind = await shownFrame();
check('overlay follows a backward seek made while paused',
  afterRewind !== null && afterRewind < whilePlaying,
  `frame ${whilePlaying} -> ${afterRewind}`);

// SEEK PAST THE LOADED WINDOW. The window is 150 frames; before the
// auto-pager the overlay simply went dead beyond it and the only way to move
// it was to type a start frame into a form.
const duration = await page.evaluate(() => document.querySelector('video.dash-video').duration);
if (duration > 6) {
  const target = Math.min(duration - 0.5, 8);
  await page.evaluate((t) => { document.querySelector('video.dash-video').currentTime = t; }, target);
  await page.waitForTimeout(2500);
  const far = await shownFrame();
  const expected = Math.floor(target * 25); // conservative lower bound
  check('overlay still has data past the 150-frame window boundary',
    far !== null && far > 150,
    `sought to ${target.toFixed(2)}s (>= frame ${expected}), overlay shows frame ${far}`);

  const markersFar = await page.evaluate(() => {
    const overlay = document.querySelector('svg.dash-video-overlay');
    return overlay ? overlay.querySelectorAll('circle').length : 0;
  });
  check('overlay still draws markers there', markersFar > 0, `${markersFar} markers`);
} else {
  console.log('   --  clip too short to cross a window boundary; skipped');
}

// PLAYBACK RATE. An independent timer would drift here; reading currentTime
// cannot.
await page.evaluate(() => {
  const video = document.querySelector('video.dash-video');
  video.currentTime = 0;
  video.playbackRate = 2;
});
await page.waitForTimeout(300);
const rateStart = await shownFrame();
await page.evaluate(() => document.querySelector('video.dash-video').play());
await page.waitForTimeout(1500);
const rateEnd = await page.evaluate(() => {
  const video = document.querySelector('video.dash-video');
  video.pause();
  return video.currentTime;
});
const rateShown = await shownFrame();
const fpsGuess = 29.97;
check('overlay stays locked to currentTime at 2x playback',
  rateShown !== null && Math.abs(rateShown - rateEnd * fpsGuess) < 25,
  `video at ${rateEnd.toFixed(2)}s (~frame ${Math.round(rateEnd * fpsGuess)}), `
  + `overlay shows frame ${rateShown} (started ${rateStart})`);

console.log('\n8. PAGE HEALTH');
check('no uncaught page errors', pageErrors.length === 0, pageErrors.join(' | '));

await browser.close();

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
