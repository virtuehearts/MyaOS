const DB_NAME = 'myaos-music-db';
const DB_VERSION = 1;
const STORE_NAME = 'track-blobs';

let dbPromise: Promise<IDBDatabase> | null = null;

function openMusicDb(): Promise<IDBDatabase> {
  if (typeof window === 'undefined' || !('indexedDB' in window)) {
    return Promise.reject(new Error('IndexedDB is not available in this environment.'));
  }

  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const request = window.indexedDB.open(DB_NAME, DB_VERSION);

      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME);
        }
      };

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error('Failed to open IndexedDB.'));
    });
  }

  return dbPromise;
}

async function withStore<T>(
  mode: IDBTransactionMode,
  run: (store: IDBObjectStore) => IDBRequest<T>
): Promise<T> {
  const db = await openMusicDb();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, mode);
    const store = transaction.objectStore(STORE_NAME);
    const request = run(store);

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB request failed.'));
  });
}

export async function putTrackBlob(key: string, blob: Blob): Promise<void> {
  await withStore('readwrite', (store) => store.put(blob, key));
}

export async function getTrackBlob(key: string): Promise<Blob | null> {
  const result = await withStore('readonly', (store) => store.get(key));
  return result ?? null;
}

export async function deleteTrackBlob(key: string): Promise<void> {
  await withStore('readwrite', (store) => store.delete(key));
}
