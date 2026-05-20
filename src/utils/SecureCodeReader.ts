import * as fs from 'fs/promises';
import * as path from 'path';
import type { CodeScope, CodeLocation } from '../models/types.js';
import { FileSystemError } from '../errors/index.js';

/**
 * SecureCodeReader provides safe file reading operations with path traversal protection.
 * All file operations are restricted to a designated project root directory.
 */
export class SecureCodeReader {
  private cache: Map<string, string> = new Map();
  private projectRoot: string;
  private readonly MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB limit
  private readonly ALLOWED_EXTENSIONS = new Set([
    '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs',
    '.py', '.java', '.go', '.rs', '.c', '.cpp', '.h',
    '.json', '.yaml', '.yml', '.toml', '.xml',
    '.md', '.txt', '.gitignore', '.env.example',
  ]);

  constructor(projectRoot: string = process.cwd()) {
    // Ensure the project root is an absolute path
    this.projectRoot = path.resolve(projectRoot);
  }

  /**
   * Validates that a file path is safe to read.
   * Prevents path traversal attacks and enforces file type restrictions.
   */
  private async validatePath(filePath: string): Promise<string> {
    // Resolve to absolute path
    const absolutePath = path.resolve(this.projectRoot, filePath);

    // Critical security check: ensure the resolved path is within project bounds
    if (!absolutePath.startsWith(this.projectRoot + path.sep)) {
      throw new FileSystemError(
        `Security violation: Path traversal attempt detected for ${filePath}`,
        'PATH_TRAVERSAL',
        filePath,
        'validate',
      );
    }

    // Check file extension
    const ext = path.extname(absolutePath).toLowerCase();
    if (ext && !this.ALLOWED_EXTENSIONS.has(ext)) {
      throw new FileSystemError(
        `Security violation: File type not allowed: ${ext}`,
        'INVALID_FILE_TYPE',
        filePath,
        'validate',
      );
    }

    // Verify file exists and check size
    try {
      const stats = await fs.stat(absolutePath);
      if (!stats.isFile()) {
        throw new FileSystemError(
          `Path is not a file: ${filePath}`,
          'NOT_A_FILE',
          filePath,
          'validate',
        );
      }
      if (stats.size > this.MAX_FILE_SIZE) {
        throw new FileSystemError(
          `File too large: ${stats.size} bytes (max: ${this.MAX_FILE_SIZE})`,
          'FILE_TOO_LARGE',
          filePath,
          'validate',
        );
      }
    } catch (error) {
      if (error instanceof FileSystemError) throw error;
      const code = (error as any).code || 'FS_ERROR';
      throw new FileSystemError(
        `Cannot access file ${filePath}: ${(error as Error).message}`,
        code,
        filePath,
        'validate',
      );
    }

    return absolutePath;
  }

  async readCodeFiles(scope: CodeScope): Promise<Map<string, string>> {
    const codeFiles = new Map<string, string>();

    // Validate and read all files in scope
    for (const file of scope.files) {
      try {
        const content = await this.readFile(file);
        // Store with the original relative path as key for consistency
        codeFiles.set(file, content);
      } catch (error) {
        console.error(`Failed to read file ${file}:`, error);
        // Continue with other files even if one fails
      }
    }

    // Read entry point files if specified
    if (scope.entryPoints) {
      for (const entryPoint of scope.entryPoints) {
        if (!codeFiles.has(entryPoint.file)) {
          try {
            const content = await this.readFile(entryPoint.file);
            codeFiles.set(entryPoint.file, content);
          } catch (error) {
            console.error(`Failed to read entry point ${entryPoint.file}:`, error);
          }
        }
      }
    }

    return codeFiles;
  }

  async readFile(filePath: string): Promise<string> {
    // Validate the path first
    const safePath = await this.validatePath(filePath);

    // Check cache using the original path
    if (this.cache.has(filePath)) {
      return this.cache.get(filePath)!;
    }

    try {
      const content = await fs.readFile(safePath, 'utf-8');
      this.cache.set(filePath, content);
      return content;
    } catch (error) {
      if (error instanceof FileSystemError) throw error;
      const code = (error as any).code || 'FS_ERROR';
      throw new FileSystemError(
        `Cannot read file ${filePath}: ${(error as Error).message}`,
        code,
        filePath,
        'read',
      );
    }
  }

  async readCodeContext(location: CodeLocation, contextLineCount: number = 50): Promise<string> {
    const content = await this.readFile(location.file);
    const lines = content.split('\n');

    const startLine = Math.max(0, location.line - contextLineCount);
    const endLine = Math.min(lines.length, location.line + contextLineCount);

    const selectedLines = lines.slice(startLine, endLine);

    // Add line numbers for clarity
    return selectedLines.map((line: string, index: number) =>
      `${startLine + index + 1}: ${line}`,
    ).join('\n');
  }

  async findRelatedFiles(baseFile: string, patterns: string[] = []): Promise<string[]> {
    // Validate the base file path first
    const safeBasePath = await this.validatePath(baseFile);

    const relatedFiles: string[] = [];
    const dir = path.dirname(safeBasePath);
    const baseName = path.basename(baseFile, path.extname(baseFile));

    try {
      const files = await fs.readdir(dir);

      for (const file of files) {
        const filePath = path.join(dir, file);

        // Skip directories and symlinks
        try {
          const stats = await fs.lstat(filePath);
          if (!stats.isFile()) {
            continue;
          }
        } catch {
          // Skip if we can't access the file
          continue;
        }

        // Convert back to relative path for consistency
        const relativePath = path.relative(this.projectRoot, filePath);

        // Skip if not a valid file type
        const ext = path.extname(file).toLowerCase();
        if (ext && !this.ALLOWED_EXTENSIONS.has(ext)) {
          continue;
        }

        // Check if it's a related file (test, spec, impl, etc.)
        if (file.includes(baseName) ||
            file.includes(`${baseName}.test`) ||
            file.includes(`${baseName}.spec`) ||
            file.includes(`${baseName}Service`) ||
            file.includes(`${baseName}Controller`)) {
          relatedFiles.push(relativePath);
        }

        // Check custom patterns
        for (const pattern of patterns) {
          if (file.includes(pattern)) {
            relatedFiles.push(relativePath);
          }
        }
      }
    } catch (error) {
      console.error(`Failed to find related files for ${baseFile}:`, error);
    }

    return relatedFiles;
  }

  /**
   * Get the configured project root directory
   */
  getProjectRoot(): string {
    return this.projectRoot;
  }

  /**
   * Update the project root (useful for testing)
   */
  setProjectRoot(newRoot: string): void {
    this.projectRoot = path.resolve(newRoot);
    this.clearCache(); // Clear cache when changing roots
  }

  clearCache(): void {
    this.cache.clear();
  }
}
