import { useCallback, useEffect, useMemo, useState } from 'react';
import { refreshScroll } from '../lib/motion.js';
import api from '../api/client.js';
import { LoadingState } from './ui/States.jsx';

import TabMatchAnalysis from './tabs/TabMatchAnalysis.jsx';
import TabPlayerIntelligence from './tabs/TabPlayerIntelligence.jsx';
import TabTeamIntelligence from './tabs/TabTeamIntelligence.jsx';
import TabCalibration from './tabs/TabCalibration.jsx';
import TabSimulation from './tabs/TabSimulation.jsx';
import TabPreMatchHealth from './tabs/TabPreMatchHealth.jsx';
import TabPsychology from './tabs/TabPsychology.jsx';
import TabCoachChat from './tabs/TabCoachChat.jsx';

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

export const TABS = [
  { id: 'match', num: '01', label: 'Match Analysis', short: 'Match', matchScoped: false },
  { id: 'player', num: '02', label: 'Player Intelligence', short: 'Player', matchScoped: true },
  { id: 'team', num: '03', label: 'Team Intelligence', short: 'Team', matchScoped: true },
  { id: 'calibration', num: '04', label: 'Calibration', short: 'Calib', matchScoped: true },
  { id: 'simulation', num: '05', label: 'Simulation', short: 'Sim', matchScoped: true },
  { id: 'health', num: '06', label: 'Pre-Match Health', short: 'Health', matchScoped: false },
  { id: 'psychology', num: '07', label: 'Pre-Match Psychology', short: 'Psych', matchScoped: false },
  { id: 'coach', num: '08', label: 'Coach Chat', short: 'Coach', matchScoped: false },
];

export const DEFAULT_TAB = 'match';

const isKnownTab = (tabId) => TABS.some((tab) => tab.id === tabId);

                                                                                
function routeNow() {
  const current = typeof window !== 'undefined' ? window.SSCView?.current?.() : null;
  if (current) return current;
                                                                          
                                                                         
                                                     
  const raw = (typeof window !== 'undefined' ? window.location.hash : '').replace(/^#\/?/, '');
  const [head, tab] = raw.split('/');
  return head === 'dashboard'
    ? { view: 'dashboard', tab: tab || null }
    : { view: 'landing', tab: null };
}

                                                                  
function matchFromUrl() {
  if (typeof window === 'undefined') return null;
  const value = new URLSearchParams(window.location.search).get('match');
  return value ? value.trim() : null;
}

                                                                                                                                                                                                                                                                                                           
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
                                                                           
                                                                          
                                                     
  const initial = routeNow();
  const [activeTab, setActiveTab] = useState(
    isKnownTab(initial.tab) ? initial.tab : DEFAULT_TAB,
  );
  const [isOpen, setIsOpen] = useState(initial.view === 'dashboard');

                                                                             
                                                                           
                                                                            
                                                                      
  const [match, setMatch] = useState({
    matchId: null,
    videoId: null,
    jobId: null,
    filename: null,
    status: null,
  });

  const matchId = match.matchId;

                                                                              
                                                                          
                                                                              
                                                                              
                                        
    
                                                                  
                                                                    
                                                                             
                                   
    
                                                                             
                                                                           
                                                                          
                                                                              
                                                                         
                                          
  const [linkedMatchId] = useState(matchFromUrl);
  const [restoring, setRestoring] = useState(() => Boolean(linkedMatchId));

  useEffect(() => {
                                                                              
                                               
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
                                                                              
                                                                            
                                              
        setRestoring(false);
        writeMatchToUrl(null);
      });

    return () => {
      active = false;
      controller.abort();
    };
                                                                           
                                                           
  }, []);

                                                                                                                                                                                                                                                                                                                                                                                  
  const goToTab = useCallback((tabId) => {
    if (!isKnownTab(tabId)) return;
    if (window.SSCView?.go) {
      window.SSCView.go(tabId);
      return;
    }
                                                                          
                                                                  
    setActiveTab(tabId);
    window.scrollTo({ top: 0, behavior: 'auto' });
  }, []);

                                                                           
                                                                               
  useEffect(() => {
    const onView = (event) => {
      const { view, tab } = event.detail || {};
      setIsOpen(view === 'dashboard');

      if (view !== 'dashboard') return;

      if (tab === null || tab === undefined) {
                                                                     
                                                   
        return;
      }

      if (isKnownTab(tab)) {
        setActiveTab(tab);
        window.scrollTo({ top: 0, behavior: 'auto' });
      } else {
                                                                          
                                                                             
                                                                          
                                              
        setActiveTab(DEFAULT_TAB);
        window.SSCView?.go?.(DEFAULT_TAB, { replace: true });
      }
    };

    window.addEventListener('ssc:view', onView);
    return () => window.removeEventListener('ssc:view', onView);
  }, []);

                                                                           
                                                                            
                                                                   
                                                                          
                                                                            
                                                                             
                                                                            
                                                

                                                                        
                                       
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

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
  const onAttachMatch = useCallback((summary) => {
    setMatch({
      matchId: summary.match_id,
                                                                          
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
                                                                                                                                                                                                  
        {restoring && active.matchScoped ? (
          <LoadingState label="Restoring match" detail="Resolving the match id in this link." rows={3} />
        ) : (
          <>
        {active.id === 'match' && <TabMatchAnalysis {...shared} />}
        {active.id === 'player' && <TabPlayerIntelligence {...shared} />}
        {active.id === 'team' && <TabTeamIntelligence {...shared} />}
        {active.id === 'calibration' && <TabCalibration {...shared} />}
        {active.id === 'simulation' && <TabSimulation {...shared} />}
        {active.id === 'health' && <TabPreMatchHealth {...shared} />}
        {active.id === 'psychology' && <TabPsychology {...shared} />}
        {active.id === 'coach' && <TabCoachChat {...shared} />}
          </>
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
