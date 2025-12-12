"use client";

import React, { useState, useRef } from "react";
import { MainLayout } from "@/components/layout/main-layout";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Loader2, Trash2, Video, MessageSquare, FileVideo, CheckCircle2, AlertCircle } from "lucide-react";
import { api, API_BASE } from "@/lib/api";
import { ChatInterface } from "@/components/conversation/chat-interface";

type IndexingResult = {
  ok: boolean;
  camera_id: number;
  clip_path: string;
  clip_url: string | null;
  indexing: Record<string, any>;
};

type VideoItem = {
  id: string;
  file: File;
  cameraId: number;
  status: 'pending' | 'uploading' | 'indexing' | 'completed' | 'error';
  result?: IndexingResult;
  error?: string;
  frames?: Array<Record<string, any>>;
};

export default function TestDashboardPage() {
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [globalEverySec, setGlobalEverySec] = useState<number>(1.0);
  const [globalWithCaptions, setGlobalWithCaptions] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState("videos");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newVideos: VideoItem[] = Array.from(e.target.files).map((file) => ({
        id: crypto.randomUUID(),
        file,
        cameraId: 99, // Default camera ID
        status: 'pending',
      }));
      setVideos((prev) => [...prev, ...newVideos]);
      // Reset input
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const removeVideo = (id: string) => {
    setVideos((prev) => prev.filter((v) => v.id !== id));
  };

  const updateVideoCameraId = (id: string, cameraId: number) => {
    setVideos((prev) => prev.map((v) => (v.id === id ? { ...v, cameraId } : v)));
  };

  const startProcessing = async () => {
    setIsProcessing(true);
    const pendingVideos = videos.filter(v => v.status === 'pending' || v.status === 'error');

    for (const video of pendingVideos) {
      // Update status to uploading
      setVideos(prev => prev.map(v => v.id === video.id ? { ...v, status: 'uploading', error: undefined } : v));

      try {
        // Upload & Index
        const res = await api.uploadVideo(video.file, {
          camera_id: video.cameraId,
          every_sec: globalEverySec,
          with_captions: globalWithCaptions,
        });

        // Update status to indexing (fetching frames)
        setVideos(prev => prev.map(v => v.id === video.id ? { ...v, status: 'indexing', result: res } : v));

        // Fetch frames
        let frames: any[] = [];
        try {
          frames = await api.listClipFrames(res.clip_path, 1000, "yolo");
        } catch (err) {
          console.error("Failed to fetch frames for", video.file.name, err);
        }

        // Complete
        setVideos(prev => prev.map(v => v.id === video.id ? { 
          ...v, 
          status: 'completed', 
          result: res,
          frames: frames || []
        } : v));

      } catch (err: any) {
        setVideos(prev => prev.map(v => v.id === video.id ? { 
          ...v, 
          status: 'error', 
          error: err?.message || String(err) 
        } : v));
      }
    }

    setIsProcessing(false);
    // Auto-switch to results if any completed
    if (videos.some(v => v.status === 'completed')) {
      setActiveTab("results");
    }
  };

  const completedCount = videos.filter(v => v.status === 'completed').length;
  const canChat = completedCount > 0;

  return (
    <MainLayout>
      <div className="h-full flex flex-col space-y-3">
        <div className="flex items-center justify-between shrink-0">
          <div>
            <h1 className="text-2xl font-bold text-foreground">Test Dashboard</h1>
            <p className="text-sm text-muted-foreground">Batch process videos and query results</p>
          </div>
          <div className="flex items-center gap-2">
             <a
                className="text-sm text-primary hover:underline"
                target="_blank"
                href={`${API_BASE.replace(/\/+$/, "")}/docs`}
                rel="noreferrer"
              >
                Backend Docs
              </a>
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col overflow-hidden">
          <TabsList className="grid w-full grid-cols-3 shrink-0">
            <TabsTrigger value="videos" className="gap-2">
              <Video className="w-4 h-4" />
              Videos
              {videos.length > 0 && <Badge variant="secondary" className="ml-1">{videos.length}</Badge>}
            </TabsTrigger>
            <TabsTrigger value="results" className="gap-2">
              <FileVideo className="w-4 h-4" />
              Indexing Results
              {completedCount > 0 && <Badge variant="secondary" className="ml-1">{completedCount}</Badge>}
            </TabsTrigger>
            <TabsTrigger value="conversation" disabled={!canChat} className="gap-2">
              <MessageSquare className="w-4 h-4" />
              Conversation
            </TabsTrigger>
          </TabsList>

          <TabsContent value="videos" className="flex-1 overflow-hidden flex flex-col gap-3 mt-3">
            <Card className="flex-1 flex flex-col overflow-hidden">
              <CardHeader className="shrink-0">
                <CardTitle>Video Queue</CardTitle>
                <CardDescription>Add videos to the queue, configure settings, and start indexing.</CardDescription>
              </CardHeader>
              <CardContent className="flex-1 overflow-hidden flex flex-col gap-3">
                {/* Global Settings */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 p-3 bg-muted/30 rounded-lg shrink-0">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Sample Every (sec)</label>
                    <input
                      type="number"
                      step="0.1"
                      min={0.1}
                      value={globalEverySec}
                      onChange={(e) => setGlobalEverySec(parseFloat(e.target.value || "1.0"))}
                      className="w-full px-3 py-2 text-sm border border-input rounded-md bg-background"
                    />
                  </div>
                  <div className="flex items-center gap-2 pt-8">
                    <input
                      id="withCaptions"
                      type="checkbox"
                      checked={globalWithCaptions}
                      onChange={(e) => setGlobalWithCaptions(e.target.checked)}
                      className="h-4 w-4"
                    />
                    <label htmlFor="withCaptions" className="text-sm text-foreground">
                      Generate captions
                    </label>
                  </div>
                  <div className="flex items-end justify-end">
                     <Button 
                        onClick={() => fileInputRef.current?.click()} 
                        variant="outline"
                        disabled={isProcessing}
                      >
                        + Add Videos
                      </Button>
                      <input
                        type="file"
                        multiple
                        accept="video/mp4"
                        ref={fileInputRef}
                        className="hidden"
                        onChange={handleFileSelect}
                      />
                  </div>
                </div>

                {/* Video List */}
                <ScrollArea className="flex-1 border rounded-md p-3">
                  {videos.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-muted-foreground opacity-50">
                      <Video className="w-12 h-12 mb-2" />
                      <p>No videos added yet</p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {videos.map((video) => (
                        <div key={video.id} className="flex items-center gap-3 p-2 border rounded-lg bg-card hover:bg-accent/5 transition-colors">
                          <div className="h-10 w-10 rounded bg-muted flex items-center justify-center shrink-0">
                            <FileVideo className="w-5 h-5 text-muted-foreground" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium truncate" title={video.file.name}>{video.file.name}</p>
                            <p className="text-xs text-muted-foreground">{Math.round(video.file.size / 1024)} KB</p>
                          </div>
                          
                          <div className="flex items-center gap-4">
                             <div className="flex flex-col">
                                <label className="text-[10px] uppercase text-muted-foreground font-bold">Camera ID</label>
                                <input
                                  type="number"
                                  min={0}
                                  value={video.cameraId}
                                  onChange={(e) => updateVideoCameraId(video.id, parseInt(e.target.value || "0", 10))}
                                  className="w-20 px-2 py-1 text-sm border border-input rounded bg-background"
                                  disabled={video.status !== 'pending' && video.status !== 'error'}
                                />
                             </div>

                             <div className="w-32 flex justify-center">
                                {video.status === 'pending' && <Badge variant="outline">Pending</Badge>}
                                {video.status === 'uploading' && <Badge className="bg-blue-500"><Loader2 className="w-3 h-3 mr-1 animate-spin" /> Uploading</Badge>}
                                {video.status === 'indexing' && <Badge className="bg-purple-500"><Loader2 className="w-3 h-3 mr-1 animate-spin" /> Indexing</Badge>}
                                {video.status === 'completed' && <Badge className="bg-green-500"><CheckCircle2 className="w-3 h-3 mr-1" /> Done</Badge>}
                                {video.status === 'error' && <Badge variant="destructive"><AlertCircle className="w-3 h-3 mr-1" /> Error</Badge>}
                             </div>

                             <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => removeVideo(video.id)}
                                disabled={isProcessing && video.status !== 'pending' && video.status !== 'error'}
                                className="text-muted-foreground hover:text-destructive"
                              >
                                <Trash2 className="w-4 h-4" />
                              </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </ScrollArea>

                <div className="flex justify-end pt-2">
                  <Button 
                    onClick={startProcessing} 
                    disabled={isProcessing || videos.filter(v => v.status === 'pending' || v.status === 'error').length === 0}
                    className="w-full md:w-auto"
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Processing Queue...
                      </>
                    ) : (
                      <>Start Indexing Queue</>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="results" className="flex-1 overflow-hidden mt-3">
             <ScrollArea className="h-full">
                <div className="space-y-6 pb-10">
                  {videos.filter(v => v.status === 'completed').length === 0 ? (
                     <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
                        <FileVideo className="w-16 h-16 mb-4 opacity-20" />
                        <p>No indexing results yet.</p>
                        <p className="text-sm">Process some videos to see results here.</p>
                     </div>
                  ) : (
                    videos.filter(v => v.status === 'completed').map((video) => (
                      <Card key={video.id}>
                        <CardHeader>
                          <div className="flex items-start justify-between">
                            <div>
                               <CardTitle className="text-base break-all">{video.file.name}</CardTitle>
                               <CardDescription>Camera ID: {video.cameraId}</CardDescription>
                            </div>
                            <Badge variant="outline" className="text-green-600 border-green-200 bg-green-50">Indexed</Badge>
                          </div>
                        </CardHeader>
                        <CardContent className="space-y-4">
                           {/* Summary Stats */}
                           <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm bg-muted/50 p-3 rounded-lg">
                              <div>
                                <span className="text-muted-foreground block text-xs">Clip Path</span>
                                <span className="font-mono text-xs break-all">{video.result?.clip_path}</span>
                              </div>
                              <div>
                                <span className="text-muted-foreground block text-xs">Clip URL</span>
                                {video.result?.clip_url ? (
                                  <a href={API_BASE.replace(/\/+$/, "") + video.result.clip_url} target="_blank" rel="noreferrer" className="text-primary hover:underline text-xs">
                                    Open Video
                                  </a>
                                ) : "N/A"}
                              </div>
                              <div>
                                <span className="text-muted-foreground block text-xs">Frames</span>
                                <span className="font-medium">{video.frames?.length || 0}</span>
                              </div>
                              <div>
                                <span className="text-muted-foreground block text-xs">Raw Data</span>
                                <span className="font-mono text-xs">{Object.keys(video.result?.indexing || {}).length} keys</span>
                              </div>
                           </div>

                           {/* Frames Table Preview */}
                           {video.frames && video.frames.length > 0 && (
                             <div className="border rounded-md overflow-hidden">
                                <div className="max-h-[300px] overflow-auto">
                                  <table className="w-full text-xs text-left">
                                    <thead className="bg-muted sticky top-0 z-10">
                                      <tr>
                                        <th className="p-2 font-medium">Idx</th>
                                        <th className="p-2 font-medium">Time</th>
                                        <th className="p-2 font-medium">Caption</th>
                                        <th className="p-2 font-medium">Objects</th>
                                      </tr>
                                    </thead>
                                    <tbody className="divide-y">
                                      {video.frames.slice(0, 50).map((fr, i) => (
                                        <tr key={i} className="hover:bg-muted/50">
                                          <td className="p-2">{fr.frame_index}</td>
                                          <td className="p-2">{fr.frame_ts}</td>
                                          <td className="p-2 max-w-[200px] truncate" title={fr.caption}>{fr.caption || "-"}</td>
                                          <td className="p-2 max-w-[150px] truncate">
                                            {Array.isArray((fr as any).object_captions) 
                                              ? ((fr as any).object_captions as string[]).join(", ") 
                                              : "-"}
                                          </td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                                {video.frames.length > 50 && (
                                  <div className="p-2 text-center text-xs text-muted-foreground bg-muted/20 border-t">
                                    Showing first 50 of {video.frames.length} frames
                                  </div>
                                )}
                             </div>
                           )}
                        </CardContent>
                      </Card>
                    ))
                  )}
                </div>
             </ScrollArea>
          </TabsContent>

          <TabsContent value="conversation" className="flex-1 overflow-hidden mt-3 flex flex-col">
             {canChat ? (
               <ChatInterface onShowSteps={() => {}} />
             ) : (
               <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground">
                  <MessageSquare className="w-16 h-16 mb-4 opacity-20" />
                  <p>Conversation unavailable.</p>
                  <p className="text-sm">Please index at least one video to start chatting.</p>
               </div>
             )}
          </TabsContent>
        </Tabs>
      </div>
    </MainLayout>
  );
}
