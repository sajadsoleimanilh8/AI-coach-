import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import DashboardApp from './components/DashboardApp.jsx';
import StreamingBackdrop from './components/ambient/StreamingBackdrop.jsx';
import './styles/dashboard.css';
import './styles/ambient.css';

                                                                                                                                                                                                                                                                                                                                                                                                                                         
const container = document.getElementById('dashboard-root');

                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
const ambient = document.getElementById('ambient-root');
if (ambient) {
  createRoot(ambient).render(
    <StrictMode>
      <StreamingBackdrop />
    </StrictMode>,
  );
}

if (container) {
  createRoot(container).render(
    <StrictMode>
      <DashboardApp />
    </StrictMode>,
  );
} else {
                                                                        
                                                                           
                                                       
  console.error(
    '[SportsStrategyCoachAI] #dashboard-root not found in index.html — the dashboard cannot mount.',
  );
}
