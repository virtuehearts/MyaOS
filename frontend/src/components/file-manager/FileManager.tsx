'use client';

import { useMemo, useState } from 'react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

const sections = [
  {
    id: 'memories',
    label: 'memories/',
    description: 'Compressed archive snapshots (zip).',
    accept: '.zip'
  },
  {
    id: 'mp3',
    label: 'mp3/',
    description: 'Audio assets for playback (mp3).',
    accept: 'audio/mpeg'
  },
  {
    id: 'backgrounds',
    label: 'backgrounds/',
    description: 'Desktop wallpapers (png, jpg).',
    accept: 'image/*'
  }
] as const;

type SectionId = (typeof sections)[number]['id'];

type FileEntry = {
  id: string;
  name: string;
  size: string;
  modified: string;
};

const fileInventory: Record<SectionId, FileEntry[]> = {
  memories: [
    {
      id: 'mem-001',
      name: 'memories-2024-09-16.zip',
      size: '18.4 MB',
      modified: 'Sep 16, 2024 · 09:12'
    },
    {
      id: 'mem-002',
      name: 'mission-log-archive.zip',
      size: '42.1 MB',
      modified: 'Aug 28, 2024 · 21:45'
    }
  ],
  mp3: [
    {
      id: 'mp3-001',
      name: 'ambient-loop.mp3',
      size: '6.2 MB',
      modified: 'Sep 10, 2024 · 18:02'
    },
    {
      id: 'mp3-002',
      name: 'status-chime.mp3',
      size: '1.1 MB',
      modified: 'Jul 30, 2024 · 13:17'
    },
    {
      id: 'mp3-003',
      name: 'system-boot-theme.mp3',
      size: '9.8 MB',
      modified: 'Jul 21, 2024 · 06:05'
    }
  ],
  backgrounds: [
    {
      id: 'bg-001',
      name: 'mya-horizon.png',
      size: '3.6 MB',
      modified: 'Sep 18, 2024 · 14:30'
    },
    {
      id: 'bg-002',
      name: 'nebula-grid.jpg',
      size: '5.9 MB',
      modified: 'Sep 08, 2024 · 08:22'
    }
  ]
};

export function FileManager() {
  const [activeSection, setActiveSection] = useState<SectionId>('memories');

  const activeConfig = useMemo(
    () => sections.find((section) => section.id === activeSection),
    [activeSection]
  );

  return (
    <div className="flex min-h-full flex-col gap-4 text-sm text-retro-text">
      <div>
        <h2 className="text-base font-semibold">File Manager</h2>
        <p className="text-xs text-retro-accent">
          Review stored assets, upload new files, and manage archives for MyaOS.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {sections.map((section) => (
          <Button
            key={section.id}
            variant={activeSection === section.id ? 'default' : 'outline'}
            size="sm"
            onClick={() => setActiveSection(section.id)}
            className="text-xs"
          >
            {section.label}
          </Button>
        ))}
      </div>

      <div className="flex flex-col gap-3 border border-retro-border bg-retro-title-active p-3">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <div className="text-xs font-semibold">{activeConfig?.label}</div>
            <p className="text-[11px] text-retro-accent">
              {activeConfig?.description}
            </p>
          </div>
          <div className="flex flex-1 flex-col items-end gap-2 text-xs">
            <label className="text-[11px] text-retro-accent">Upload file</label>
            <Input
              type="file"
              accept={activeConfig?.accept}
              className="h-8 max-w-[220px] text-[11px]"
            />
          </div>
        </div>

        <div className="grid gap-2 text-xs">
          <div className="hidden grid-cols-[minmax(0,1fr)_90px_150px_170px] gap-2 border-b border-retro-border/60 pb-2 text-[11px] uppercase text-retro-accent md:grid">
            <span>File</span>
            <span>Size</span>
            <span>Last modified</span>
            <span>Actions</span>
          </div>
          {fileInventory[activeSection].map((file) => (
            <div
              key={file.id}
              className="grid gap-2 border border-retro-border bg-retro-surface p-2 md:grid-cols-[minmax(0,1fr)_90px_150px_170px]"
            >
              <div>
                <div className="font-semibold text-retro-text">{file.name}</div>
                <div className="mt-1 text-[11px] text-retro-accent md:hidden">
                  {file.size} · {file.modified}
                </div>
              </div>
              <div className="hidden md:block">{file.size}</div>
              <div className="hidden md:block">{file.modified}</div>
              <div className="flex flex-wrap gap-2">
                <Button variant="ghost" size="sm" className="h-7 px-2 text-[10px]">
                  Download
                </Button>
                <Button variant="ghost" size="sm" className="h-7 px-2 text-[10px]">
                  Rename
                </Button>
                <Button variant="ghost" size="sm" className="h-7 px-2 text-[10px]">
                  Delete
                </Button>
              </div>
            </div>
          ))}
          {fileInventory[activeSection].length === 0 && (
            <div className="text-[11px] text-retro-accent">
              No files yet. Upload a file to get started.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
