/**
 * Shared time-formatting utilities used across the dashboard.
 */

/** Human-readable relative time (e.g. "3m ago", "2h ago"). */
export function timeAgo(ts: string): string {
    const d = new Date(ts);
    const diffMs = Date.now() - d.getTime();
    const mins = Math.floor(diffMs / 60_000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.floor(hrs / 24)}d ago`;
}
