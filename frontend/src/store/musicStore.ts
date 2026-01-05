import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

export type PlaybackMode = 'normal' | 'repeat' | 'loop';

export interface TrackMetadata {
  id: string;
  name: string;
  duration: number | null;
  order: number;
  blobKey: string;
}

interface MusicStoreState {
  tracks: TrackMetadata[];
  currentTrackId: string | null;
  isPlaying: boolean;
  playbackMode: PlaybackMode;
  volume: number;
  addTrack: (track: TrackMetadata) => void;
  removeTrack: (trackId: string) => void;
  reorderTracks: (orderedIds: string[]) => void;
  setCurrentTrack: (trackId: string | null) => void;
  setIsPlaying: (isPlaying: boolean) => void;
  setPlaybackMode: (mode: PlaybackMode) => void;
  setVolume: (volume: number) => void;
  updateTrackDuration: (trackId: string, duration: number) => void;
  getNextTrackId: (direction: 1 | -1) => string | null;
}

const storage =
  typeof window === 'undefined'
    ? undefined
    : createJSONStorage(() => window.localStorage);

export const useMusicStore = create<MusicStoreState>()(
  persist(
    (set, get) => ({
      tracks: [],
      currentTrackId: null,
      isPlaying: false,
      playbackMode: 'normal',
      volume: 0.8,
      addTrack: (track) =>
        set((state) => {
          const nextTracks = [...state.tracks, track].sort((a, b) => a.order - b.order);
          const nextCurrent = state.currentTrackId ?? track.id;
          return { tracks: nextTracks, currentTrackId: nextCurrent };
        }),
      removeTrack: (trackId) =>
        set((state) => {
          const nextTracks = state.tracks.filter((track) => track.id !== trackId);
          const nextCurrent =
            state.currentTrackId === trackId
              ? nextTracks[0]?.id ?? null
              : state.currentTrackId;
          return { tracks: nextTracks, currentTrackId: nextCurrent };
        }),
      reorderTracks: (orderedIds) =>
        set((state) => {
          const orderMap = new Map(orderedIds.map((id, index) => [id, index]));
          const nextTracks = state.tracks
            .map((track) => ({
              ...track,
              order: orderMap.get(track.id) ?? track.order
            }))
            .sort((a, b) => a.order - b.order);
          return { tracks: nextTracks };
        }),
      setCurrentTrack: (trackId) => set({ currentTrackId: trackId }),
      setIsPlaying: (isPlaying) => set({ isPlaying }),
      setPlaybackMode: (mode) => set({ playbackMode: mode }),
      setVolume: (volume) => set({ volume: Math.min(1, Math.max(0, volume)) }),
      updateTrackDuration: (trackId, duration) =>
        set((state) => ({
          tracks: state.tracks.map((track) =>
            track.id === trackId ? { ...track, duration } : track
          )
        })),
      getNextTrackId: (direction: 1 | -1) => {
        const { tracks, currentTrackId } = get();
        if (tracks.length === 0) {
          return null;
        }
        const currentIndex = tracks.findIndex((track) => track.id === currentTrackId);
        const baseIndex = currentIndex === -1 ? 0 : currentIndex;
        const nextIndex = (baseIndex + direction + tracks.length) % tracks.length;
        return tracks[nextIndex]?.id ?? null;
      }
    }),
    {
      name: 'mya-music-store',
      storage,
      partialize: (state) => ({
        tracks: state.tracks,
        currentTrackId: state.currentTrackId,
        isPlaying: state.isPlaying,
        playbackMode: state.playbackMode,
        volume: state.volume
      })
    }
  )
);
