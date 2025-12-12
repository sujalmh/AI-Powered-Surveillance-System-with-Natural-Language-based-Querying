export type ChatSendResponse = {
    session_id: string;
    parsed_filter: Record<string, any>;
    results: Array<{
        camera_id: number;
        timestamp?: string;
        start?: string;
        end?: string;
        clip_url?: string;
        score?: number;
        source?: string;
        objects?: Array<{
            object_name: string;
            track_id: number;
            confidence: number;
            color?: string;
            person_attributes?: Record<string, any>;
        }>;
    }>;
    answer: string;
    metadata?: Record<string, any>;
    alert_created?: Record<string, any>;
};

export const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "") || "http://localhost:8000";

async function http<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(`${API_BASE}${path}`, {
        ...init,
        headers: {
            "Content-Type": "application/json",
            ...(init?.headers || {}),
        },
        cache: "no-store",
    });
    if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status} ${res.statusText}: ${text}`);
    }
    return res.json() as Promise<T>;
}

export const api = {
    health: () => http<{ status: string; version: string }>("/health"),

    // Chat
    chatSend: (sessionId: string, message: string, limit = 50) =>
        http<ChatSendResponse>("/api/chat/send", {
            method: "POST",
            body: JSON.stringify({
                session_id: sessionId,
                message,
                limit,
            }),
        }),

    chatHistory: (sessionId: string) =>
        http<{ session_id: string; messages: Array<{ role: string; content: string; created_at: string }> }>(
            `/api/chat/history?session_id=${encodeURIComponent(sessionId)}`
        ),
    chatSessions: () =>
        http<Array<{ session_id: string; last_message?: string; last_message_time?: string; message_count: number }>>(
            "/api/chat/sessions"
        ),

    // Cameras
    listCameras: () => http<Array<Record<string, any>>>("/api/cameras/"),
    registerCamera: (payload: { camera_id: number; source: number | string; location?: string }) =>
        http<{ ok: boolean; camera_id: number }>("/api/cameras/register", {
            method: "POST",
            body: JSON.stringify(payload),
        }),
    startCamera: (cameraId: number, payload?: { source?: number | string; location?: string; show_window?: boolean }) =>
        http<{ ok: boolean; camera_id: number; running: boolean }>(`/api/cameras/${cameraId}/start`, {
            method: "POST",
            body: JSON.stringify(payload || {}),
        }),
    stopCamera: (cameraId: number) =>
        http<{ ok: boolean; camera_id: number; running: boolean }>(`/api/cameras/${cameraId}/stop`, {
            method: "POST",
        }),
    probeCamera: (source: string | number, timeout = 3) => {
        const src = typeof source === "number" ? String(source) : source;
        const p = new URLSearchParams();
        p.set("source", src);
        p.set("timeout", String(timeout));
        return http<{ ok: boolean; message: string }>(`/api/cameras/probe?${p.toString()}`);
    },

    // Detections
    listDetections: (params: URLSearchParams) =>
        http<Array<Record<string, any>>>(`/api/detections/?${params.toString()}`),
    objectCounts: (opts?: { camera_id?: number; from?: string; to?: string; last_minutes?: number; top_n?: number }) => {
        const p = new URLSearchParams();
        if (opts?.camera_id != null) p.set("camera_id", String(opts.camera_id));
        if (opts?.from) p.set("from", opts.from);
        if (opts?.to) p.set("to", opts.to);
        if (opts?.last_minutes != null) p.set("last_minutes", String(opts.last_minutes));
        if (opts?.top_n != null) p.set("top_n", String(opts.top_n));
        return http<Array<{ object_name: string; count: number }>>(`/api/detections/object-counts?${p.toString()}`);
    },
    colorCounts: (opts?: { camera_id?: number; object?: string; from?: string; to?: string; last_minutes?: number; top_n?: number }) => {
        const p = new URLSearchParams();
        if (opts?.camera_id != null) p.set("camera_id", String(opts.camera_id));
        if (opts?.object) p.set("object", opts.object);
        if (opts?.from) p.set("from", opts.from);
        if (opts?.to) p.set("to", opts.to);
        if (opts?.last_minutes != null) p.set("last_minutes", String(opts.last_minutes));
        if (opts?.top_n != null) p.set("top_n", String(opts.top_n));
        return http<Array<{ color: string; count: number }>>(`/api/detections/colors?${p.toString()}`);
    },

    // Alerts
    listAlerts: () => http<Array<Record<string, any>>>("/api/alerts/"),
    createAlert: (payload: Record<string, any>) =>
        http<Record<string, any>>("/api/alerts/", { method: "POST", body: JSON.stringify(payload) }),
    evaluateAlerts: () => http<Record<string, any>>("/api/alerts/evaluate", { method: "POST" }),
    listAlertLogs: () => http<Array<Record<string, any>>>("/api/alerts/logs"),
    deleteAlert: (alertId: string) =>
        http<{ ok: boolean }>(`/api/alerts/${alertId}`, { method: "DELETE" }),
    evaluateAlert: (alertId: string) =>
        http<Record<string, any>>(`/api/alerts/${alertId}/evaluate`, { method: "POST" }),

    // Alerts SSE stream (Server-Sent Events)
    streamAlerts: (opts?: { camera_id?: number; last_ts?: string }) => {
        const p = new URLSearchParams();
        if (opts?.camera_id != null) p.set("camera_id", String(opts.camera_id));
        if (opts?.last_ts) p.set("last_ts", opts.last_ts);
        const url = `${API_BASE}/api/alerts/stream?${p.toString()}`;
        return new EventSource(url);
    },

    // Videos/media
    listRecordings: () => http<Array<Record<string, any>>>("/api/videos/recordings"),
    listClips: () => http<Array<Record<string, any>>>("/api/videos/clips"),
    listSnapshots: () => http<Array<Record<string, any>>>("/api/videos/snapshots"),

    // Frame-level details for a specific clip
    listClipFrames: (clip_path: string, limit: number = 500, enrich?: "yolo") => {
        const p = new URLSearchParams();
        p.set("clip_path", clip_path);
        p.set("limit", String(limit));
        if (enrich) p.set("enrich", enrich);
        return http<Array<Record<string, any>>>(`/api/videos/frames?${p.toString()}`);
    },

    // Video upload + indexing
    uploadVideo: async (
        file: File,
        opts?: { camera_id?: number; every_sec?: number; with_captions?: boolean }
    ): Promise<{
        ok: boolean;
        camera_id: number;
        clip_path: string;
        clip_url: string | null;
        indexing: Record<string, any>;
    }> => {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("camera_id", String(opts?.camera_id ?? 99));
        fd.append("every_sec", String(opts?.every_sec ?? 1.0));
        fd.append("with_captions", String(!!opts?.with_captions));

        const res = await fetch(`${API_BASE}/api/videos/upload`, {
            method: "POST",
            body: fd,
            // note: do NOT set Content-Type; browser sets multipart boundary automatically
            cache: "no-store",
        });
        if (!res.ok) {
            const text = await res.text().catch(() => "");
            throw new Error(`HTTP ${res.status} ${res.statusText}: ${text}`);
        }
        return res.json();
    },

    // Settings
    getIndexingMode: () =>
        http<{ indexing_mode: "structured" | "semantic" | "both" }>("/api/settings/indexing-mode"),
    setIndexingMode: (mode: "structured" | "semantic" | "both") =>
        http<{ indexing_mode: "structured" | "semantic" | "both" }>("/api/settings/indexing-mode", {
            method: "PUT",
            body: JSON.stringify({ indexing_mode: mode }),
        }),
    resetSystem: () => http<{ status: string; message: string }>("/api/settings/reset", { method: "DELETE" }),
};
