import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { TaskbarMusicPlayer } from '@/components/os/TaskbarMusicPlayer';
import { getTrackBlob } from '@/lib/musicDb';
import { useMusicStore } from '@/store/musicStore';

vi.mock('@/lib/musicDb', () => ({
  getTrackBlob: vi.fn(),
  putTrackBlob: vi.fn(),
  deleteTrackBlob: vi.fn()
}));

describe('TaskbarMusicPlayer', () => {
  const playSpy = HTMLMediaElement.prototype.play as unknown as ReturnType<typeof vi.fn>;
  const pauseSpy = HTMLMediaElement.prototype.pause as unknown as ReturnType<typeof vi.fn>;

  beforeEach(() => {
    useMusicStore.setState({
      tracks: [
        {
          id: 'track-1',
          name: 'Test Track.mp3',
          duration: null,
          order: 0,
          blobKey: 'track-1'
        }
      ],
      currentTrackId: 'track-1',
      isPlaying: false,
      playbackMode: 'normal',
      volume: 0.8
    });
    (getTrackBlob as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Blob(['test-audio'], { type: 'audio/mpeg' })
    );
    playSpy.mockClear();
    pauseSpy.mockClear();
  });

  afterEach(() => {
    useMusicStore.setState({
      tracks: [],
      currentTrackId: null,
      isPlaying: false,
      playbackMode: 'normal',
      volume: 0.8
    });
    useMusicStore.persist.clearStorage();
  });

  it('loads and plays audio with basic playback controls', async () => {
    const { container } = render(<TaskbarMusicPlayer />);

    await waitFor(() => {
      expect(getTrackBlob).toHaveBeenCalled();
    });

    const audio = container.querySelector('audio') as HTMLAudioElement;
    await waitFor(() => {
      expect(audio.src).toContain('blob:mock');
    });

    fireEvent.click(screen.getByLabelText('Play'));
    await waitFor(() => {
      expect(playSpy).toHaveBeenCalled();
      expect(useMusicStore.getState().isPlaying).toBe(true);
    });

    audio.currentTime = 12;
    expect(audio.currentTime).toBe(12);

    fireEvent.click(screen.getByLabelText('Pause'));
    await waitFor(() => {
      expect(pauseSpy).toHaveBeenCalled();
      expect(useMusicStore.getState().isPlaying).toBe(false);
    });

    fireEvent.click(screen.getByLabelText('Play'));
    await waitFor(() => {
      expect(useMusicStore.getState().isPlaying).toBe(true);
    });

    audio.dispatchEvent(new Event('ended'));
    await waitFor(() => {
      expect(useMusicStore.getState().isPlaying).toBe(false);
    });
  });
});
