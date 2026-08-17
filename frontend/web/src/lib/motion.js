                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

function motion() {
  return typeof window !== 'undefined' ? window.SSCMotion : undefined;
}

                                                                                                                                                                                   
export function animateIn(root) {
  motion()?.animateIn?.(root);
}

                                                                            
export function refreshScroll() {
  motion()?.refresh?.();
}

                                                                                                                                                     
export function countUp(target, onValue) {
  const bridge = motion();
  if (bridge?.countUp) return bridge.countUp(target, onValue);

                                                                            
                                                     
  const value = Number(target);
  onValue(Number.isFinite(value) ? String(value) : null);
  return () => {};
}
