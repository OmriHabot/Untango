/**
 * Local Directory Utilities
 * 
 * Handles browser-based directory access using the File System Access API.
 * Provides functionality to pick directories, scan files, create zip bundles,
 * and compute file hashes for watch mode.
 */

import JSZip from 'jszip';

// File extensions to include (matching the CLI tool)
const INCLUDE_EXTENSIONS = new Set([
  '.py', '.js', '.ts', '.tsx', '.jsx', '.md', '.txt', 
  '.json', '.yaml', '.yml', '.html', '.css', '.ipynb'
]);

// Directories to skip
// Note: .git is NOT skipped - we need it for git tools to work
// Note: .venv and venv are NOT skipped - we want to include virtual environments
const SKIP_DIRS = new Set([
  '__pycache__', 'node_modules', 
  '.repos', 'dist', 'build', '.next', '.pytest_cache', 
  '.mypy_cache', 'egg-info', '.tox', '.coverage'
]);

export interface ScannedFile {
  path: string;
  file: File;
}

export interface ScanResult {
  files: ScannedFile[];
  totalSize: number;
  directoryName: string;
}

/**
 * Check if the File System Access API is supported
 */
export function isFileSystemAccessSupported(): boolean {
  return 'showDirectoryPicker' in window;
}

/**
 * Prompt user to pick a directory
 */
export async function pickDirectory(): Promise<FileSystemDirectoryHandle | null> {
  if (!isFileSystemAccessSupported()) {
    console.warn('File System Access API is not supported in this browser');
    return null;
  }

  try {
    // @ts-expect-error - showDirectoryPicker may not be in TypeScript's lib
    const handle = await window.showDirectoryPicker({
      mode: 'read'
    });
    return handle;
  } catch (err: unknown) {
    // User cancelled the picker
    if (err instanceof Error && err.name === 'AbortError') {
      return null;
    }
    throw err;
  }
}

/**
 * Check if a file extension should be included
 */
function shouldIncludeFile(filename: string): boolean {
  const ext = filename.substring(filename.lastIndexOf('.')).toLowerCase();
  return INCLUDE_EXTENSIONS.has(ext);
}

/**
 * Check if a directory should be skipped
 */
function shouldSkipDirectory(name: string): boolean {
  // Explicitly skip these directories
  if (SKIP_DIRS.has(name)) {
    return true;
  }
  
  // Hidden directories that we WANT to include
  const INCLUDE_HIDDEN_DIRS = new Set(['.git', '.venv', '.env']);
  
  // Skip hidden directories by default, except those we explicitly want
  if (name.startsWith('.') && !INCLUDE_HIDDEN_DIRS.has(name)) {
    return true;
  }
  
  return false;
}

/**
 * Recursively scan a directory and collect files
 * @param isInsideGitDir - If true, include ALL files (for .git directory)
 */
export async function scanDirectory(
  dirHandle: FileSystemDirectoryHandle,
  basePath: string = '',
  isInsideGitDir: boolean = false
): Promise<ScannedFile[]> {
  const files: ScannedFile[] = [];

  for await (const entry of dirHandle.values()) {
    const entryPath = basePath ? `${basePath}/${entry.name}` : entry.name;

    if (entry.kind === 'directory') {
      if (!shouldSkipDirectory(entry.name)) {
        const subHandle = await dirHandle.getDirectoryHandle(entry.name);
        // Track if we're entering a .git directory
        const enteringGitDir = entry.name === '.git' || isInsideGitDir;
        const subFiles = await scanDirectory(subHandle, entryPath, enteringGitDir);
        files.push(...subFiles);
      }
    } else if (entry.kind === 'file') {
      // Include all files inside .git, otherwise check extension
      if (isInsideGitDir || shouldIncludeFile(entry.name)) {
        const fileHandle = await dirHandle.getFileHandle(entry.name);
        const file = await fileHandle.getFile();
        files.push({ path: entryPath, file });
      }
    }
  }

  return files;
}

/**
 * Scan directory and return summary
 */
export async function scanDirectoryWithSummary(
  dirHandle: FileSystemDirectoryHandle
): Promise<ScanResult> {
  const files = await scanDirectory(dirHandle);
  const totalSize = files.reduce((sum, f) => sum + f.file.size, 0);

  return {
    files,
    totalSize,
    directoryName: dirHandle.name
  };
}

/**
 * Create a zip bundle from scanned files
 */
export async function createZipBundle(files: ScannedFile[]): Promise<Blob> {
  const zip = new JSZip();

  for (const { path, file } of files) {
    const content = await file.arrayBuffer();
    zip.file(path, content);
  }

  return await zip.generateAsync({ 
    type: 'blob',
    compression: 'DEFLATE',
    compressionOptions: { level: 6 }
  });
}

/**
 * Compute MD5-like hash for a file (using SubtleCrypto)
 * Note: Uses SHA-256 instead of MD5 since SubtleCrypto doesn't support MD5
 */
async function computeFileHash(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Compute hashes for all files in a scan result
 */
export async function computeFileHashes(
  files: ScannedFile[]
): Promise<Record<string, string>> {
  const hashes: Record<string, string> = {};

  for (const { path, file } of files) {
    hashes[path] = await computeFileHash(file);
  }

  return hashes;
}

/**
 * Compare two hash maps and return changed/new files
 */
export function findChangedFiles(
  oldHashes: Record<string, string>,
  newHashes: Record<string, string>
): string[] {
  const changed: string[] = [];

  for (const [path, newHash] of Object.entries(newHashes)) {
    const oldHash = oldHashes[path];
    if (oldHash !== newHash) {
      changed.push(path);
    }
  }

  return changed;
}

/**
 * Format file size for display
 */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
