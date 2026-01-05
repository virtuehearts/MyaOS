'use client';

import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react';

import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { deleteTrackBlob, getTrackBlob, putTrackBlob } from '@/lib/musicDb';
import { useMusicStore } from '@/store/musicStore';

const formatTime = (value: number) => {
  if (!Number.isFinite(value)) {
    return '--:--';
  }
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60);
  return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
};

const getSafeId = () => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `track-${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

export function TaskbarMusicPlayer() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  const {
    tracks,
    currentTrackId,
    isPlaying,
    playbackMode,
    volume,
    addTrack,
    removeTrack,
    reorderTracks,
    setCurrentTrack,
    setIsPlaying,
    setPlaybackMode,
    setVolume,
    updateTrackDuration
  } = useMusicStore();

  const orderedTracks = useMemo(
    () => [...tracks].sort((a, b) => a.order - b.order),
    [tracks]
  );

  const currentTrack = orderedTracks.find((track) => track.id === currentTrackId) ?? null;

  const loadCurrentTrack = useCallback(async () => {
    const audio = audioRef.current;
    if (!audio || !currentTrack) {
      if (audio) {
        audio.removeAttribute('src');
        audio.load();
      }
      setCurrentTime(0);
      setDuration(0);
      setIsPlaying(false);
      return;
    }

    setIsLoading(true);
    try {
      const blob = await getTrackBlob(currentTrack.blobKey);
      if (!blob) {
        setIsLoading(false);
        setIsPlaying(false);
        return;
      }
      const objectUrl = URL.createObjectURL(blob);
      audio.src = objectUrl;
      audio.load();
      setIsLoading(false);
      if (isPlaying) {
        void audio.play().catch(() => setIsPlaying(false));
      }
      return () => URL.revokeObjectURL(objectUrl);
    } catch (error) {
      setIsLoading(false);
      setIsPlaying(false);
    }
    return undefined;
  }, [currentTrack, isPlaying, setIsPlaying]);

  useEffect(() => {
    const cleanupPromise = loadCurrentTrack();
    return () => {
      void cleanupPromise?.then((cleanup) => cleanup?.());
    };
  }, [loadCurrentTrack]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    audio.volume = volume;
  }, [volume]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
    };
    const handleMetadata = () => {
      setDuration(Number.isFinite(audio.duration) ? audio.duration : 0);
      if (currentTrack && Number.isFinite(audio.duration)) {
        updateTrackDuration(currentTrack.id, audio.duration);
      }
    };
    const handleEnded = () => {
      if (!currentTrack) {
        setIsPlaying(false);
        return;
      }
      if (playbackMode === 'repeat') {
        audio.currentTime = 0;
        void audio.play();
        return;
      }
      const currentIndex = orderedTracks.findIndex((track) => track.id === currentTrack.id);
      const nextIndex =
        playbackMode === 'loop'
          ? (currentIndex + 1 + orderedTracks.length) % orderedTracks.length
          : currentIndex + 1;
      const nextTrack = orderedTracks[nextIndex];
      if (!nextTrack) {
        setIsPlaying(false);
        return;
      }
      setCurrentTrack(nextTrack.id);
    };

    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('loadedmetadata', handleMetadata);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('loadedmetadata', handleMetadata);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [
    currentTrack,
    orderedTracks,
    playbackMode,
    setCurrentTrack,
    setIsPlaying,
    updateTrackDuration
  ]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    if (isPlaying) {
      void audio.play().catch(() => setIsPlaying(false));
    } else {
      audio.pause();
    }
  }, [isPlaying, setIsPlaying]);

  const handlePlayToggle = () => {
    if (!currentTrack && orderedTracks.length > 0) {
      setCurrentTrack(orderedTracks[0].id);
    }
    setIsPlaying(!isPlaying);
  };

  const handlePrev = () => {
    if (orderedTracks.length === 0) {
      return;
    }
    const currentIndex = orderedTracks.findIndex((track) => track.id === currentTrackId);
    const nextIndex =
      currentIndex <= 0 ? orderedTracks.length - 1 : (currentIndex - 1) % orderedTracks.length;
    setCurrentTrack(orderedTracks[nextIndex]?.id ?? orderedTracks[0]?.id ?? null);
  };

  const handleNext = () => {
    if (orderedTracks.length === 0) {
      return;
    }
    const currentIndex = orderedTracks.findIndex((track) => track.id === currentTrackId);
    const nextIndex = (currentIndex + 1) % orderedTracks.length;
    setCurrentTrack(orderedTracks[nextIndex]?.id ?? orderedTracks[0]?.id ?? null);
  };

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    if (files.length === 0) {
      return;
    }
    const baseOrder = orderedTracks.length;
    await Promise.all(
      files.map(async (file, index) => {
        const id = getSafeId();
        const blobKey = `track-${id}`;
        await putTrackBlob(blobKey, file);
        addTrack({
          id,
          name: file.name,
          duration: null,
          order: baseOrder + index,
          blobKey
        });
      })
    );
    event.target.value = '';
  };

  const handleRemove = async (trackId: string, blobKey: string) => {
    await deleteTrackBlob(blobKey);
    removeTrack(trackId);
  };

  const handleMove = (trackId: string, direction: -1 | 1) => {
    const index = orderedTracks.findIndex((track) => track.id === trackId);
    const targetIndex = index + direction;
    if (index === -1 || targetIndex < 0 || targetIndex >= orderedTracks.length) {
      return;
    }
    const nextOrder = [...orderedTracks];
    const [item] = nextOrder.splice(index, 1);
    nextOrder.splice(targetIndex, 0, item);
    reorderTracks(nextOrder.map((track) => track.id));
  };

  const handleProgressChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextTime = Number(event.target.value);
    const audio = audioRef.current;
    if (audio) {
      audio.currentTime = nextTime;
      setCurrentTime(nextTime);
    }
  };

  const handleVolumeChange = (event: ChangeEvent<HTMLInputElement>) => {
    setVolume(Number(event.target.value));
  };

  const handlePlaybackModeToggle = () => {
    setPlaybackMode(
      playbackMode === 'normal' ? 'repeat' : playbackMode === 'repeat' ? 'loop' : 'normal'
    );
  };

  return (
    <div className="relative flex items-center gap-2">
      <audio ref={audioRef} />
      <Button
        variant="ghost"
        size="sm"
        onClick={handlePrev}
        aria-label="Previous track"
        className="h-8 px-2"
      >
        ◀
      </Button>
      <Button
        variant="ghost"
        size="sm"
        onClick={handlePlayToggle}
        aria-label={isPlaying ? 'Pause' : 'Play'}
        className="h-8 px-3"
        disabled={orderedTracks.length === 0 && !currentTrackId}
      >
        {isLoading ? '...' : isPlaying ? 'Pause' : 'Play'}
      </Button>
      <Button
        variant="ghost"
        size="sm"
        onClick={handleNext}
        aria-label="Next track"
        className="h-8 px-2"
      >
        ▶
      </Button>
      <Button
        variant="ghost"
        size="sm"
        onClick={handlePlaybackModeToggle}
        aria-label="Toggle playback mode"
        className={cn('h-8 px-2', playbackMode !== 'normal' && 'bg-retro-title-active')}
        title={`Playback mode: ${playbackMode}`}
      >
        {playbackMode === 'repeat' ? 'Repeat' : playbackMode === 'loop' ? 'Loop' : 'Normal'}
      </Button>
      <div className="flex w-32 items-center gap-2">
        <input
          type="range"
          min={0}
          max={duration || 0}
          step={1}
          value={Math.min(currentTime, duration || 0)}
          onChange={handleProgressChange}
          className="h-1 w-full cursor-pointer accent-retro-accent"
          aria-label="Playback progress"
        />
        <span className="min-w-[56px] text-[10px]">
          {formatTime(currentTime)} / {formatTime(duration || 0)}
        </span>
      </div>
      <div className="flex items-center gap-1">
        <span className="text-[10px]">Vol</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={volume}
          onChange={handleVolumeChange}
          className="h-1 w-16 cursor-pointer accent-retro-accent"
          aria-label="Volume"
        />
      </div>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setIsPanelOpen((open) => !open)}
        aria-label="Open playlist"
        className="h-8 px-2"
      >
        Playlist
      </Button>
      {isPanelOpen ? (
        <div className="absolute bottom-12 right-0 z-50 w-80 space-y-3 rounded border border-retro-border bg-retro-surface p-3 shadow-[0_8px_0_rgba(0,0,0,0.15)]">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-retro-text">Playlist</span>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-2 text-[10px]"
              onClick={() => setIsPanelOpen(false)}
            >
              Close
            </Button>
          </div>
          <label className="flex flex-col gap-2 text-[11px]">
            Upload MP3s
            <input
              type="file"
              accept="audio/mpeg"
              multiple
              onChange={handleUpload}
              className="w-full text-[11px]"
            />
          </label>
          <div className="max-h-48 space-y-2 overflow-y-auto pr-1 text-[11px]">
            {orderedTracks.length === 0 ? (
              <p className="text-[11px] text-retro-text/70">No tracks yet.</p>
            ) : (
              orderedTracks.map((track, index) => (
                <div
                  key={track.id}
                  className={cn(
                    'flex items-center justify-between gap-2 rounded border border-retro-border px-2 py-1',
                    currentTrackId === track.id && 'bg-retro-title-active'
                  )}
                >
                  <button
                    type="button"
                    className="flex flex-1 flex-col text-left"
                    onClick={() => setCurrentTrack(track.id)}
                  >
                    <span className="truncate font-semibold">{track.name}</span>
                    <span className="text-[10px] text-retro-text/70">
                      {track.duration ? formatTime(track.duration) : '--:--'}
                    </span>
                  </button>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 px-2 text-[10px]"
                      onClick={() => handleMove(track.id, -1)}
                      disabled={index === 0}
                    >
                      ↑
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 px-2 text-[10px]"
                      onClick={() => handleMove(track.id, 1)}
                      disabled={index === orderedTracks.length - 1}
                    >
                      ↓
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 px-2 text-[10px]"
                      onClick={() => handleRemove(track.id, track.blobKey)}
                    >
                      ✕
                    </Button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}
