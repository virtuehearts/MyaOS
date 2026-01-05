'use client';

import { useEffect, useMemo, useState, type ChangeEvent, type MouseEvent } from 'react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { FileContextMenu, type ContextMenuAction } from '@/components/file-manager/FileContextMenu';
import { TextEditor } from '@/components/file-manager/TextEditor';
import { API_BASE_URL, apiRequest } from '@/lib/api';
import { useAuthStore } from '@/store/authStore';

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
  },
  {
    id: 'notes',
    label: 'notes/',
    description: 'Plain-text and markdown notes (.txt, .md).',
    accept: '.txt,.md'
  }
] as const;

type SectionId = (typeof sections)[number]['id'];

type FileEntry = {
  name: string;
  size: number;
  modified: string;
};

const textExtensions = ['.txt', '.md'];

const formatBytes = (bytes: number) => {
  if (Number.isNaN(bytes)) {
    return '--';
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const units = ['KB', 'MB', 'GB'];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(1)} ${units[unitIndex]}`;
};

const formatDate = (value: string) => {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return '--';
  }
  return date.toLocaleString('en-US', {
    month: 'short',
    day: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
};

const isTextFile = (filename: string) => {
  const lower = filename.toLowerCase();
  return textExtensions.some((ext) => lower.endsWith(ext));
};

export function FileManager() {
  const [activeSection, setActiveSection] = useState<SectionId>('memories');
  const { token } = useAuthStore();
  const [entries, setEntries] = useState<FileEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [contextMenu, setContextMenu] = useState<{
    isOpen: boolean;
    position: { x: number; y: number };
    target: FileEntry | null;
  }>({
    isOpen: false,
    position: { x: 0, y: 0 },
    target: null
  });
  const [editorOpen, setEditorOpen] = useState(false);
  const [editorContent, setEditorContent] = useState('');
  const [editorPath, setEditorPath] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const activeConfig = useMemo(
    () => sections.find((section) => section.id === activeSection),
    [activeSection]
  );

  const loadFiles = async (bucket = activeSection) => {
    if (!token) {
      setEntries([]);
      return;
    }
    setIsLoading(true);
    try {
      const response = await apiRequest<{ entries: FileEntry[] }>(
        `/files?bucket=${bucket}`,
        {
          headers: { Authorization: `Bearer ${token}` }
        }
      );
      setEntries(response.entries);
    } catch (error) {
      console.error(error);
      setEntries([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadFiles(activeSection);
  }, [activeSection, token]);

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !token) {
      return;
    }
    setUploadError(null);
    const formData = new FormData();
    formData.append('bucket', activeSection);
    formData.append('file', file);
    try {
      const response = await fetch(`${API_BASE_URL}/files/upload`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData
      });
      if (!response.ok) {
        let detail = 'Upload failed.';
        try {
          const payload = await response.json();
          detail = payload.detail ?? detail;
        } catch (parseError) {
          const fallback = await response.text();
          if (fallback) {
            detail = fallback;
          }
          console.error(parseError);
        }
        throw new Error(detail);
      }
      await loadFiles(activeSection);
    } catch (error) {
      console.error(error);
      setUploadError(
        error instanceof Error ? error.message : 'Unable to upload that file.'
      );
    } finally {
      event.target.value = '';
    }
  };

  const handleNewFile = async () => {
    if (!token || activeSection !== 'notes') {
      return;
    }
    const rawName = window.prompt('Enter a file name (without extension).');
    if (!rawName) {
      return;
    }
    const extension = window
      .prompt('Choose an extension: txt or md', 'txt')
      ?.trim()
      .replace('.', '')
      .toLowerCase();
    if (!extension || !['txt', 'md'].includes(extension)) {
      window.alert('Please choose either .txt or .md.');
      return;
    }
    const filename = `${rawName}.${extension}`;
    try {
      await apiRequest('/files/text', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          bucket: activeSection,
          path: filename,
          content: ''
        })
      });
      setEditorPath(filename);
      setEditorContent('');
      setEditorOpen(true);
      await loadFiles(activeSection);
    } catch (error) {
      console.error(error);
      window.alert('Unable to create that file.');
    }
  };

  const handleOpenText = async (file: FileEntry) => {
    if (!token || !isTextFile(file.name)) {
      return;
    }
    try {
      const response = await apiRequest<{ content: string }>(
        `/files/text?bucket=${activeSection}&path=${encodeURIComponent(file.name)}`,
        {
          headers: { Authorization: `Bearer ${token}` }
        }
      );
      setEditorPath(file.name);
      setEditorContent(response.content ?? '');
      setEditorOpen(true);
    } catch (error) {
      console.error(error);
      window.alert('Unable to open that file.');
    }
  };

  const handleSaveText = async () => {
    if (!token || !editorPath) {
      return;
    }
    setIsSaving(true);
    try {
      await apiRequest('/files/text', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          bucket: activeSection,
          path: editorPath,
          content: editorContent
        })
      });
      await loadFiles(activeSection);
      setEditorOpen(false);
    } catch (error) {
      console.error(error);
      window.alert('Unable to save that file.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = async (file: FileEntry) => {
    if (!token) {
      return;
    }
    const confirmed = window.confirm(`Delete ${file.name}?`);
    if (!confirmed) {
      return;
    }
    try {
      await apiRequest(
        `/files?bucket=${activeSection}&path=${encodeURIComponent(file.name)}`,
        {
          method: 'DELETE',
          headers: { Authorization: `Bearer ${token}` }
        }
      );
      await loadFiles(activeSection);
    } catch (error) {
      console.error(error);
      window.alert('Unable to delete that file.');
    }
  };

  const openContextMenu = (event: MouseEvent, target: FileEntry | null) => {
    event.preventDefault();
    setContextMenu({
      isOpen: true,
      position: { x: event.clientX, y: event.clientY },
      target
    });
  };

  const contextActions: ContextMenuAction[] = useMemo(() => {
    const target = contextMenu.target;
    return [
      {
        id: 'new-file',
        label: 'New File',
        onSelect: handleNewFile,
        disabled: activeSection !== 'notes'
      },
      {
        id: 'open',
        label: 'Open',
        onSelect: () => {
          if (target) {
            handleOpenText(target);
          }
        },
        disabled: !target || !isTextFile(target.name)
      },
      {
        id: 'delete',
        label: 'Delete',
        onSelect: () => {
          if (target) {
            handleDelete(target);
          }
        },
        disabled: !target
      }
    ];
  }, [contextMenu.target, activeSection, handleNewFile, handleOpenText, handleDelete]);

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
              onChange={handleUpload}
            />
            {uploadError && (
              <p className="max-w-[220px] text-[11px] text-red-400">
                {uploadError}
              </p>
            )}
          </div>
        </div>

        <div
          className="grid gap-2 text-xs"
          onContextMenu={(event) => openContextMenu(event, null)}
        >
          <div className="hidden grid-cols-[minmax(0,1fr)_90px_150px_170px] gap-2 border-b border-retro-border/60 pb-2 text-[11px] uppercase text-retro-accent md:grid">
            <span>File</span>
            <span>Size</span>
            <span>Last modified</span>
            <span>Actions</span>
          </div>
          {entries.map((file) => (
            <div
              key={file.name}
              className="grid gap-2 border border-retro-border bg-retro-surface p-2 md:grid-cols-[minmax(0,1fr)_90px_150px_170px]"
              onClick={() => handleOpenText(file)}
              onContextMenu={(event) => openContextMenu(event, file)}
            >
              <div>
                <div className="font-semibold text-retro-text">{file.name}</div>
                <div className="mt-1 text-[11px] text-retro-accent md:hidden">
                  {formatBytes(file.size)} · {formatDate(file.modified)}
                </div>
              </div>
              <div className="hidden md:block">{formatBytes(file.size)}</div>
              <div className="hidden md:block">{formatDate(file.modified)}</div>
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-[10px]"
                  onClick={(event) => {
                    event.stopPropagation();
                    handleOpenText(file);
                  }}
                  disabled={!isTextFile(file.name)}
                >
                  Open
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-[10px]"
                  onClick={(event) => event.stopPropagation()}
                >
                  Rename
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-[10px]"
                  onClick={(event) => {
                    event.stopPropagation();
                    handleDelete(file);
                  }}
                >
                  Delete
                </Button>
              </div>
            </div>
          ))}
          {!isLoading && entries.length === 0 && (
            <div className="text-[11px] text-retro-accent">
              No files yet. Upload a file to get started.
            </div>
          )}
          {isLoading && (
            <div className="text-[11px] text-retro-accent">Loading files…</div>
          )}
        </div>
      </div>

      <FileContextMenu
        isOpen={contextMenu.isOpen}
        position={contextMenu.position}
        actions={contextActions}
        onClose={() => setContextMenu((prev) => ({ ...prev, isOpen: false }))}
      />

      <TextEditor
        isOpen={editorOpen}
        filename={editorPath}
        content={editorContent}
        isSaving={isSaving}
        onChange={setEditorContent}
        onClose={() => setEditorOpen(false)}
        onSave={handleSaveText}
      />
    </div>
  );
}
