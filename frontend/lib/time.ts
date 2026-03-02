/**
 * Shared time-formatting utilities used across the dashboard.
 */

import { format, toZonedTime } from "date-fns-tz";

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

/**
 * Format timestamp to IST (Asia/Kolkata) timezone.
 * Returns format: "Mar 2, 2026 14:30 IST"
 * Handles null/invalid timestamps gracefully.
 */
export function formatToIST(ts: string | null | undefined): string {
    if (!ts) return "—";
    
    try {
        // Parse the timestamp (assumed to be UTC if no timezone info)
        const date = new Date(ts);
        
        // Check if date is valid
        if (isNaN(date.getTime())) return "Invalid date";
        
        // Convert to IST (Asia/Kolkata) - IST is UTC+5:30
        const istDate = toZonedTime(date, "Asia/Kolkata");
        
        // Format: "Mar 2, 2026 14:30 IST" (abbreviated month, no seconds)
        return format(istDate, "MMM d, yyyy HH:mm", { timeZone: "Asia/Kolkata" }) + " IST";
    } catch (error) {
        console.error("Error formatting timestamp to IST:", error);
        return "—";
    }
}

/**
 * Format time-only in IST (Asia/Kolkata) timezone.
 * Returns format: "14:30 IST"
 * Used for contexts where only time is needed (e.g., message timestamps).
 */
export function formatTimeIST(ts: string | null | undefined): string {
    if (!ts) return "—";
    
    try {
        const date = new Date(ts);
        if (isNaN(date.getTime())) return "Invalid time";
        
        const istDate = toZonedTime(date, "Asia/Kolkata");
        return format(istDate, "HH:mm", { timeZone: "Asia/Kolkata" }) + " IST";
    } catch (error) {
        console.error("Error formatting time to IST:", error);
        return "—";
    }
}
