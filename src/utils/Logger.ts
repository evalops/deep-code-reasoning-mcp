/**
 * Logger utility for MCP server
 *
 * MCP servers using stdio transport MUST log to stderr (not stdout)
 * because stdout is reserved for MCP protocol communication.
 * This logger provides semantic methods while ensuring correct output stream.
 */
export class Logger {
  private prefix: string;

  constructor(prefix: string = '[MCP]') {
    this.prefix = prefix;
  }

  /**
   * Log informational messages to stderr
   */
  info(...args: any[]): void {
    console.error(this.prefix, ...args);
  }

  /**
   * Log warning messages to stderr
   */
  warn(...args: any[]): void {
    console.error(`${this.prefix} [WARN]`, ...args);
  }

  /**
   * Log error messages to stderr
   */
  error(...args: any[]): void {
    console.error(`${this.prefix} [ERROR]`, ...args);
  }

  /**
   * Log debug messages to stderr (only if DEBUG env var is set)
   */
  debug(...args: any[]): void {
    if (process.env.DEBUG) {
      console.error(`${this.prefix} [DEBUG]`, ...args);
    }
  }
}

// Default logger instance
export const logger = new Logger();