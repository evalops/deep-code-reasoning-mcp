import { z } from 'zod';
import type { ClaudeCodeContext, Finding } from '../models/types.js';

/**
 * InputValidator provides schema validation for user inputs to prevent malicious data
 */
export class InputValidator {
  // Safe string schema with length limits
  static readonly SafeString = z.string()
    .max(2000)
    .regex(/^[^<>{}]*$/, 'String contains potentially unsafe characters');

  // Safe filename schema
  static readonly SafeFilename = z.string()
    .max(255)
    .regex(/^[a-zA-Z0-9._\-/]+$/, 'Invalid filename format')
    .refine(path => !path.includes('..'), 'Path traversal detected');

  // Array of safe strings
  static readonly SafeStringArray = z.array(this.SafeString).max(100);

  // Array of safe filenames
  static readonly SafeFilenameArray = z.array(this.SafeFilename).max(100);

  // Finding schema with validation
  static readonly FindingSchema = z.object({
    type: z.enum(['bug', 'performance', 'architecture', 'security']),
    severity: z.enum(['low', 'medium', 'high', 'critical']),
    location: z.object({
      file: this.SafeFilename,
      line: z.number().min(0).max(100000),
      column: z.number().min(0).max(1000).optional(),
      functionName: this.SafeString.optional(),
    }),
    description: this.SafeString,
    evidence: this.SafeStringArray,
  });

  // Partial findings array
  static readonly PartialFindingsSchema = z.array(this.FindingSchema).max(50);

  // Code scope schema
  static readonly CodeScopeSchema = z.object({
    files: this.SafeFilenameArray,
    entryPoints: z.array(z.object({
      file: this.SafeFilename,
      line: z.number().min(0).max(100000),
      column: z.number().min(0).max(1000).optional(),
      functionName: this.SafeString.optional(),
    })).max(20).optional(),
    serviceNames: this.SafeStringArray.optional(),
    searchPatterns: this.SafeStringArray.optional(),
  });

  // Claude context schema
  static readonly ClaudeContextSchema = z.object({
    attempted_approaches: this.SafeStringArray,
    partial_findings: z.array(z.any()).max(50), // Will be validated separately
    stuck_description: this.SafeString,
    code_scope: this.CodeScopeSchema,
  });

  /**
   * Validate and sanitize ClaudeCodeContext
   */
  static validateClaudeContext(input: any): ClaudeCodeContext {
    // First validate the basic structure
    const validated = this.ClaudeContextSchema.parse(input);

    // Validate partial findings separately
    const validatedFindings: Finding[] = [];
    for (const finding of validated.partial_findings) {
      try {
        const validFinding = this.FindingSchema.parse(finding);
        validatedFindings.push(validFinding);
      } catch {
        console.warn('Invalid finding skipped:', finding);
      }
    }

    return {
      attemptedApproaches: validated.attempted_approaches,
      partialFindings: validatedFindings,
      stuckPoints: [validated.stuck_description], // Convert to array format
      focusArea: {
        files: validated.code_scope.files,
        entryPoints: validated.code_scope.entryPoints || [],
        serviceNames: validated.code_scope.serviceNames,
        searchPatterns: validated.code_scope.searchPatterns,
      },
      analysisBudgetRemaining: 60, // Default value
    };
  }

  /**
   * Validate a single string input
   */
  static validateString(input: unknown, maxLength: number = 2000): string {
    if (typeof input !== 'string') {
      throw new Error('Input must be a string');
    }

    return this.SafeString.parse(input.substring(0, maxLength));
  }

  /**
   * Validate an array of strings
   */
  static validateStringArray(input: unknown, maxItems: number = 100): string[] {
    if (!Array.isArray(input)) {
      return [];
    }

    return input
      .slice(0, maxItems)
      .filter(item => typeof item === 'string')
      .map(item => this.validateString(item));
  }

  /**
   * Validate file paths
   */
  static validateFilePaths(paths: unknown): string[] {
    if (!Array.isArray(paths)) {
      return [];
    }

    const validPaths: string[] = [];
    for (const path of paths) {
      try {
        const validated = this.SafeFilename.parse(path);
        validPaths.push(validated);
      } catch {
        console.warn(`Invalid file path rejected: ${path}`);
      }
    }

    return validPaths;
  }
}
