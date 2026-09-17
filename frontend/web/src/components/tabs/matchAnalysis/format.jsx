

/* ==================================================================== */

export function formatTime(value) {
  if (!value) return <span className="dash-absent">—</span>;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return <span className="dash-absent">—</span>;
  return date.toLocaleString();
}
