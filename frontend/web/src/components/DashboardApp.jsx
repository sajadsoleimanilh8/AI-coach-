import { useCallback, useEffect, useMemo, useState } from 'react';
import { refreshScroll } from '../lib/motion.js';
import api from '../api/client.js';
import { ErrorBoundary } from './ui/ErrorBoundary.jsx';
import { LoadingState } from './ui/States.jsx';

import TabMatchAnalysis from './tabs/TabMatchAnalysis.jsx';
import TabPlayerIntelligence from './tabs/TabPlayerIntelligence.jsx';
import TabTeamIntelligence from './tabs/TabTeamIntelligence.jsx';
import TabCalibration from './tabs/TabCalibration.jsx';
import TabSimulation from './tabs/TabSimulation.jsx';
import TabPreMatchHealth from './tabs/TabPreMatchHealth.jsx';
import TabPsychology from './tabs/TabPsychology.jsx';
import TabCoachChat from './tabs/TabCoachChat.jsx';

/**
 * The dashboard's root. Mounted into #dashboard-root inside the existing
 * index.html -- there is no second page and no router; index.html's own
 * script toggles body[data-view] between "landing" and "dashboard" and this
 * component listens for that event.
 *
 * Shared state lives here and only here: `matchId` is set once Match
 * Analysis completes an upload, and every match-scoped tab reads it from
 * props. A tab never derives a match id of its own.
 */

const TABS = [
  { id: 'match', num: '01', label: 'Match Analysis', short: 'Match', matchScoped: false },
  { id: 'player', num: '02', label: 'Player Intelligence', short: 'Player', matchScoped: true },
  { id: 'team', num: '03', label: 'Team Intelligence', short: 'Team', matchScoped: true },
  { id: 'calibration', num: '04', label: 'Calibration', short: 'Calib', matchScoped: true },
  { id: 'simulation', num: '05', label: 'Simulation', short: 'Sim', matchScoped: true },
  { id: 'health', num: '06', label: 'Pre-Match Health', short: 'Health', matchScoped: false },
  { id: 'psychology', num: '07', label: 'Pre-Match Psychology', short: 'Psych', matchScoped: false },
  { id: 'coach', num: '08', label: 'Coach Chat', short: 'Coach', matchScoped: false },
];

const DEFAULT_TAB = 'match';

const isKnownTab = (tabId) => TABS.some((tab) => tab.id === tabId);

/** Reads the current route directly, so first paint already matches the URL. */
function routeNow() {
  const current = typeof window !== 'undefined' ? window.SSCView?.current?.() : null;
  if (current) return current;
  // index.html has not run yet (or was edited to drop SSCView). Parse the
  // same hash shape rather than silently defaulting to the landing view,
  // which would blank a deep link on a hard refresh.
  const raw = (typeof window !== 'undefined' ? window.location.hash : '').replace(/^#\/?/, '');
  const [head, tab] = raw.split('/');
  return head === 'dashboard'
    ? { view: 'dashboard', tab: tab || null }
    : { view: 'landing', tab: null };
}

/** The selected match id carried in the query string, or null. */
function matchFromUrl() {
  if (typeof window === 'undefined') return null;
  const value = new URLSearchParams(window.location.search).get('match');
  return value ? value.trim() : null;
}

/**
 * Writes (or clears) ?match=<id> without touching the hash route.
 *
 * replaceState, not pushState: selecting a match is not a navigation, and
 * pushing here would make Back undo the selection instead of returning to
 * the previous tab -- the "back button does something surprising" bug.
 */
function writeMatchToUrl(matchId) {
  if (typeof window === 'undefined') return;
  const params = new URLSearchParams(window.location.search);
  if (matchId) params.set('match', matchId);
  else params.delete('match');
  const query = params.toString();
  const url = window.location.pathname + (query ? `?${query}` : '') + window.location.hash;
  window.history.replaceState(window.history.state, '', url);
}

export default function DashboardApp() {
  // Initialised FROM the route, not from a hardcoded default: a refresh on
  // #dashboard/simulation must render Simulation on the very first paint,
  // never Match Analysis followed by a visible swap.
  const initial = routeNow();
  const [activeTab, setActiveTab] = useState(
    isKnownTab(initial.tab) ? initial.tab : DEFAULT_TAB,
  );
  const [isOpen, setIsOpen] = useState(initial.view === 'dashboard');

  // --- shared match state -------------------------------------------------
  // Lifted out of Match Analysis on a successful upload. `job` and `video`
  // ride along because the latency report is keyed by job_id and the player
  // is keyed by video_id, and re-deriving either would mean guessing.
  const [match, setMatch] = useState({
    matchId: null,
    videoId: null,
    jobId: null,
    filename: null,
    status: null,
  });

  const matchId = match.matchId;

  // The selection is held in the URL as ?match=<id>, so it survives a refresh
  // and can be linked to. It used to live only in this component's state,
  // which meant a reload -- or opening a deep link like #dashboard/simulation
  // directly -- dropped it, and the four match-scoped tabs came up empty with
  // no way back except retyping a UUID.
  //
  // Only the id is stored. video_id / job_id are re-resolved from
  // GET /api/matches/{id} rather than carried in the URL, because a
  // hand-edited or stale link must not be able to pair one match's clip with
  // another match's tracking data.
  //
  // Captured during the FIRST RENDER, before any effect can run. Reading the
  // URL inside the restore effect instead meant the writer effect below --
  // which is registered first, so it runs first, and on mount sees a null
  // matchId -- had already stripped ?match= from the address bar. The restore
  // then found nothing to restore and every refresh silently dropped the
  // selection it was written to preserve.
  const [linkedMatchId] = useState(matchFromUrl);
  const [restoring, setRestoring] = useState(() => Boolean(linkedMatchId));

  useEffect(() => {
    // Never write while restoring: matchId is still null then, and writing it
    // would delete the very id being resolved.
    if (restoring) return;
    writeMatchToUrl(matchId);
  }, [matchId, restoring]);

  useEffect(() => {
    const fromUrl = linkedMatchId;
    if (!fromUrl) return undefined;

    const controller = new AbortController();
    let active = true;

    api.getMatchSummary(fromUrl, controller.signal)
      .then((summary) => {
        if (!active) return;
        setMatch({
          matchId: summary.match_id,
          videoId: summary.video_file_exists ? summary.video_id : null,
          jobId: summary.job_id ?? null,
          filename: summary.video_filename ?? null,
          status: summary.job_status ?? null,
        });
        setRestoring(false);
      })
      .catch((error) => {
        if (!active || error?.name === 'AbortError') return;
        // A link to a match that no longer exists must not leave a dead id in
        // the address bar pretending to be a selection. Drop it and let the
        // picker show what is actually there.
        setRestoring(false);
        writeMatchToUrl(null);
      });

    return () => {
      active = false;
      controller.abort();
    };
    // Runs once: later changes to ?match= are this component's own writes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /**
   * Switching tabs goes through the URL, never straight to setState.
   *
   * SSCView.go() pushes the history entry and then dispatches 'ssc:view',
   * which the subscription below turns into setActiveTab. One direction of
   * flow — click -> URL -> state — is what keeps the address bar, the back
   * button, and the rendered tab from ever disagreeing.
   */
  const goToTab = useCallback((tabId) => {
    if (!isKnownTab(tabId)) return;
    if (window.SSCView?.go) {
      window.SSCView.go(tabId);
      return;
    }
    // No router available (index.html's script missing). Still switch the
    // tab so the console stays usable; only deep-linking is lost.
    setActiveTab(tabId);
    window.scrollTo({ top: 0, behavior: 'auto' });
  }, []);

  // index.html owns the route; this is the subscription that turns a route
  // change — click, deep link, refresh, or back/forward — into rendered state.
  useEffect(() => {
    const onView = (event) => {
      const { view, tab } = event.detail || {};
      setIsOpen(view === 'dashboard');

      if (view !== 'dashboard') return;

      if (tab === null || tab === undefined) {
        // Bare #dashboard: keep whichever tab is showing rather than
        // yanking the user back to Match Analysis.
        return;
      }

      if (isKnownTab(tab)) {
        setActiveTab(tab);
        window.scrollTo({ top: 0, behavior: 'auto' });
      } else {
        // An unknown tab id in the URL (typo, stale bookmark). Render the
        // default AND correct the address bar, so what is displayed and what
        // is in the URL cannot disagree. replace, so Back does not bounce
        // straight back onto the broken link.
        setActiveTab(DEFAULT_TAB);
        window.SSCView?.go?.(DEFAULT_TAB, { replace: true });
      }
    };

    window.addEventListener('ssc:view', onView);
    return () => window.removeEventListener('ssc:view', onView);
  }, []);

  // Unlike the landing nav (initNavBehavior in index.html), this header is
  // the app's only way to switch tabs, not decorative wayfinding -- it does
  // NOT hide on scroll. A prior version mirrored the landing nav's
  // hide-on-scroll behaviour here and it made every tab unreachable after
  // scrolling any tab's content: the header would translate off-screen and,
  // since #dashboard's content routinely scrolls past the y > 100 threshold,
  // stay hidden until the user scrolled back to the very top -- which looks
  // exactly like "clicking a tab does nothing".

  // A tab swap changes the page height, which ScrollTrigger has already
  // measured for the landing sections.
  useEffect(() => {
    refreshScroll();
  }, [activeTab, isOpen]);

  const onUploadComplete = useCallback((upload) => {
    setMatch({
      matchId: upload.match_id,
      videoId: upload.video_id,
      jobId: upload.job_id,
      filename: upload.filename,
      status: upload.status,
    });
  }, []);

  /**
   * Adopting a match that already exists in the database, from the backend's
   * own summary of it (GET /api/matches/{match_id}).
   *
   * Every id here comes from that response. Any that the backend reports as
   * absent is stored as null rather than inherited from a previous upload --
   * carrying a stale video_id across would play one match's clip underneath
   * another match's tracking overlay, which is exactly the class of quietly
   * wrong reading this dashboard exists to prevent.
   */
  const onAttachMatch = useCallback((summary) => {
    setMatch({
      matchId: summary.match_id,
      // A Video row whose file is gone from disk is not a playable video.
      videoId: summary.video_file_exists ? summary.video_id : null,
      jobId: summary.job_id ?? null,
      filename: summary.video_filename ?? null,
      status: summary.job_status ?? null,
    });
  }, []);

  const shared = useMemo(
    () => ({ ...match, goToTab, onUploadComplete, onAttachMatch }),
    [match, goToTab, onUploadComplete, onAttachMatch],
  );

  const exit = useCallback(() => {
    window.SSCView?.close?.();
  }, []);

  const active = TABS.find((tab) => tab.id === activeTab) || TABS[0];

  return (
    <section id="dashboard" aria-label="Tactical console" aria-hidden={!isOpen}>
      <header className="dash-header">
        <div className="dash-header-top">
          <div className="dash-brand">
            <span className="nav-logo">
              STRATEGY<span>.AI</span>
            </span>
            <span className="dash-brand-sub">// Tactical Console</span>
          </div>

          <div className="dash-header-right">
            <MatchChip matchId={matchId} filename={match.filename} />
            <button type="button" className="nav-cta dash-exit" onClick={exit}>
              ← Overview
            </button>
          </div>
        </div>

        <nav className="dash-tabs" aria-label="Dashboard sections">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={`dash-tab ${tab.id === activeTab ? 'is-active' : ''}`.trim()}
              aria-current={tab.id === activeTab ? 'page' : undefined}
              onClick={() => goToTab(tab.id)}
            >
              <span className="dash-tab-num">{tab.num}</span>
              <span className="dash-tab-label">{tab.label}</span>
              <span className="dash-tab-short">{tab.short}</span>
              {tab.matchScoped && !matchId ? (
                <span className="dash-tab-dot" title="Needs a match" />
              ) : null}
            </button>
          ))}
        </nav>
      </header>

      <main className="dash-main">
        {/* Restoring ?match= is a real load, so a match-scoped tab says so
            instead of flashing "Select a Match First" at a user who has in
            fact already selected one. */}
        {restoring && active.matchScoped ? (
          <LoadingState label="Restoring match" detail="Resolving the match id in this link." rows={3} />
        ) : (
          // Keyed by tab id so switching tabs always starts from a clean
          // boundary: a crash in one tab never follows the user to the next.
          <ErrorBoundary key={active.id}>
        {active.id === 'match' && <TabMatchAnalysis {...shared} />}
        {/* Match-scoped tabs are keyed by match, so switching match remounts
            them with fresh state (selection, interventions, in-flight requests)
            instead of each tab resetting itself from an effect. */}
        {active.id === 'player' && <TabPlayerIntelligence key={matchId} {...shared} />}
        {active.id === 'team' && <TabTeamIntelligence key={matchId} {...shared} />}
        {active.id === 'calibration' && <TabCalibration {...shared} />}
        {active.id === 'simulation' && <TabSimulation key={matchId} {...shared} />}
        {active.id === 'health' && <TabPreMatchHealth {...shared} />}
        {active.id === 'psychology' && <TabPsychology {...shared} />}
        {active.id === 'coach' && <TabCoachChat {...shared} />}
          </ErrorBoundary>
        )}
      </main>

      <footer className="dash-footer">
        <span className="footer-meta">
          Core API :8000 · NEXUS :8100 · Every number on this screen is served by one of them.
        </span>
        <span className="footer-meta">
          System Status: <span className="text-accent-purple">◉ Console Active</span>
        </span>
      </footer>
    </section>
  );
}

function MatchChip({ matchId, filename }) {
  if (!matchId) {
    return (
      <span className="dash-match-chip is-empty" title="No match uploaded in this session yet">
        <span className="dash-match-chip-key">Match</span>
        None
      </span>
    );
  }
  return (
    <span className="dash-match-chip" title={filename ? `From ${filename}` : matchId}>
      <span className="dash-match-chip-key">Match</span>
      <code>{matchId}</code>
    </span>
  );
}
