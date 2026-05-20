#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import * as dotenv from 'dotenv';

import { DeepCodeReasonerV2 } from './analyzers/DeepCodeReasonerV2.js';
import type { ClaudeCodeContext } from './models/types.js';
import { ErrorClassifier } from './utils/ErrorClassifier.js';
import { InputValidator } from './utils/InputValidator.js';
import { logger } from './utils/Logger.js';

// Load environment variables
dotenv.config();

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const server = new Server(
  {
    name: 'deep-code-reasoning-mcp',
    version: '0.1.0',
  },
  {
    capabilities: {
      tools: {},
    },
  },
);

// Initialize deepReasoner as null if no API key
let deepReasoner: DeepCodeReasonerV2 | null = null;
if (GEMINI_API_KEY) {
  deepReasoner = new DeepCodeReasonerV2(GEMINI_API_KEY);
}

const EscalateAnalysisSchema = z.object({
  claude_context: z.object({
    attempted_approaches: z.array(z.string()),
    partial_findings: z.array(z.any()),
    stuck_description: z.string(),
    code_scope: z.object({
      files: z.array(z.string()),
      entry_points: z.array(z.any()).optional(),
      service_names: z.array(z.string()).optional(),
    }),
  }),
  analysis_type: z.enum(['execution_trace', 'cross_system', 'performance', 'hypothesis_test']),
  depth_level: z.number().min(1).max(5),
  time_budget_seconds: z.number().default(60),
});

const TraceExecutionPathSchema = z.object({
  entry_point: z.object({
    file: z.string(),
    line: z.number(),
    function_name: z.string().optional(),
  }),
  max_depth: z.number().default(10),
  include_data_flow: z.boolean().default(true),
});

const HypothesisTestSchema = z.object({
  hypothesis: z.string(),
  code_scope: z.object({
    files: z.array(z.string()),
    entry_points: z.array(z.any()).optional(),
  }),
  test_approach: z.string(),
});

const CrossSystemImpactSchema = z.object({
  change_scope: z.object({
    files: z.array(z.string()),
    service_names: z.array(z.string()).optional(),
  }),
  impact_types: z.array(z.enum(['breaking', 'performance', 'behavioral'])).optional(),
});

const PerformanceBottleneckSchema = z.object({
  code_path: z.object({
    entry_point: z.object({
      file: z.string(),
      line: z.number(),
      function_name: z.string().optional(),
    }),
    suspected_issues: z.array(z.string()).optional(),
  }),
  profile_depth: z.number().min(1).max(5).default(3),
});

const StartConversationSchema = z.object({
  claude_context: z.object({
    attempted_approaches: z.array(z.string()),
    partial_findings: z.array(z.any()),
    stuck_description: z.string(),
    code_scope: z.object({
      files: z.array(z.string()),
      entry_points: z.array(z.any()).optional(),
      service_names: z.array(z.string()).optional(),
    }),
  }),
  analysis_type: z.enum(['execution_trace', 'cross_system', 'performance', 'hypothesis_test']),
  initial_question: z.string().optional(),
});

const ContinueConversationSchema = z.object({
  session_id: z.string(),
  message: z.string(),
  include_code_snippets: z.boolean().optional(),
});

const FinalizeConversationSchema = z.object({
  session_id: z.string(),
  summary_format: z.enum(['detailed', 'concise', 'actionable']).optional(),
});

const GetConversationStatusSchema = z.object({
  session_id: z.string(),
});

const RunHypothesisTournamentSchema = z.object({
  claude_context: z.object({
    attempted_approaches: z.array(z.string()),
    partial_findings: z.array(z.any()),
    stuck_description: z.string(),
    code_scope: z.object({
      files: z.array(z.string()),
      entry_points: z.array(z.any()).optional(),
      service_names: z.array(z.string()).optional(),
    }),
  }),
  issue: z.string(),
  tournament_config: z.object({
    max_hypotheses: z.number().min(2).max(20).optional(),
    max_rounds: z.number().min(1).max(5).optional(),
    parallel_sessions: z.number().min(1).max(10).optional(),
  }).optional(),
});

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: 'escalate_analysis',
        description: 'Hand off complex analysis to Gemini when Claude Code hits reasoning limits. Gemini will perform deep semantic analysis beyond syntactic patterns.',
        inputSchema: {
          type: 'object',
          properties: {
            claude_context: {
              type: 'object',
              properties: {
                attempted_approaches: {
                  type: 'array',
                  items: { type: 'string' },
                  description: 'What Claude Code already tried',
                },
                partial_findings: {
                  type: 'array',
                  description: 'Any findings Claude Code discovered',
                },
                stuck_description: {
                  type: 'string',
                  description: 'Description of where Claude Code got stuck',
                },
                code_scope: {
                  type: 'object',
                  properties: {
                    files: {
                      type: 'array',
                      items: { type: 'string' },
                      description: 'Files to analyze',
                    },
                    entry_points: {
                      type: 'array',
                      description: 'Specific functions/methods to start from',
                    },
                    service_names: {
                      type: 'array',
                      items: { type: 'string' },
                      description: 'Services involved in cross-system analysis',
                    },
                  },
                  required: ['files'],
                },
              },
              required: ['attempted_approaches', 'partial_findings', 'stuck_description', 'code_scope'],
            },
            analysis_type: {
              type: 'string',
              enum: ['execution_trace', 'cross_system', 'performance', 'hypothesis_test'],
              description: 'Type of deep analysis to perform',
            },
            depth_level: {
              type: 'number',
              minimum: 1,
              maximum: 5,
              description: 'How deep to analyze (1=shallow, 5=very deep)',
            },
            time_budget_seconds: {
              type: 'number',
              default: 60,
              description: 'Maximum time for analysis',
            },
          },
          required: ['claude_context', 'analysis_type'],
        },
      },
      {
        name: 'trace_execution_path',
        description: 'Use Gemini to perform deep execution analysis with semantic understanding',
        inputSchema: {
          type: 'object',
          properties: {
            entry_point: {
              type: 'object',
              properties: {
                file: { type: 'string' },
                line: { type: 'number' },
                function_name: { type: 'string' },
              },
              required: ['file', 'line'],
            },
            max_depth: { type: 'number', default: 10 },
            include_data_flow: { type: 'boolean', default: true },
          },
          required: ['entry_point'],
        },
      },
      {
        name: 'hypothesis_test',
        description: 'Use Gemini to test specific theories about code behavior',
        inputSchema: {
          type: 'object',
          properties: {
            hypothesis: { type: 'string' },
            code_scope: {
              type: 'object',
              properties: {
                files: { type: 'array', items: { type: 'string' } },
                entry_points: { type: 'array' },
              },
              required: ['files'],
            },
            test_approach: { type: 'string' },
          },
          required: ['hypothesis', 'code_scope', 'test_approach'],
        },
      },
      {
        name: 'cross_system_impact',
        description: 'Use Gemini to analyze changes across service boundaries',
        inputSchema: {
          type: 'object',
          properties: {
            change_scope: {
              type: 'object',
              properties: {
                files: { type: 'array', items: { type: 'string' } },
                service_names: { type: 'array', items: { type: 'string' } },
              },
              required: ['files'],
            },
            impact_types: {
              type: 'array',
              items: { type: 'string', enum: ['breaking', 'performance', 'behavioral'] },
            },
          },
          required: ['change_scope'],
        },
      },
      {
        name: 'performance_bottleneck',
        description: 'Use Gemini for deep performance analysis with execution modeling',
        inputSchema: {
          type: 'object',
          properties: {
            code_path: {
              type: 'object',
              properties: {
                entry_point: {
                  type: 'object',
                  properties: {
                    file: { type: 'string' },
                    line: { type: 'number' },
                    function_name: { type: 'string' },
                  },
                  required: ['file', 'line'],
                },
                suspected_issues: { type: 'array', items: { type: 'string' } },
              },
              required: ['entry_point'],
            },
            profile_depth: { type: 'number', minimum: 1, maximum: 5, default: 3 },
          },
          required: ['code_path'],
        },
      },
      {
        name: 'start_conversation',
        description: 'Start a conversational analysis session between Claude and Gemini',
        inputSchema: {
          type: 'object',
          properties: {
            claude_context: {
              type: 'object',
              properties: {
                attempted_approaches: {
                  type: 'array',
                  items: { type: 'string' },
                  description: 'What Claude Code already tried',
                },
                partial_findings: {
                  type: 'array',
                  description: 'Any findings Claude Code discovered',
                },
                stuck_description: {
                  type: 'string',
                  description: 'Description of where Claude Code got stuck',
                },
                code_scope: {
                  type: 'object',
                  properties: {
                    files: {
                      type: 'array',
                      items: { type: 'string' },
                      description: 'Files to analyze',
                    },
                    entry_points: {
                      type: 'array',
                      description: 'Specific functions/methods to start from',
                    },
                    service_names: {
                      type: 'array',
                      items: { type: 'string' },
                      description: 'Services involved in cross-system analysis',
                    },
                  },
                  required: ['files'],
                },
              },
              required: ['attempted_approaches', 'partial_findings', 'stuck_description', 'code_scope'],
            },
            analysis_type: {
              type: 'string',
              enum: ['execution_trace', 'cross_system', 'performance', 'hypothesis_test'],
              description: 'Type of deep analysis to perform',
            },
            initial_question: {
              type: 'string',
              description: 'Initial question to start the conversation',
            },
          },
          required: ['claude_context', 'analysis_type'],
        },
      },
      {
        name: 'continue_conversation',
        description: 'Continue an ongoing analysis conversation',
        inputSchema: {
          type: 'object',
          properties: {
            session_id: {
              type: 'string',
              description: 'ID of the conversation session',
            },
            message: {
              type: 'string',
              description: 'Claude\'s response or follow-up question',
            },
            include_code_snippets: {
              type: 'boolean',
              description: 'Whether to include code snippets in response',
            },
          },
          required: ['session_id', 'message'],
        },
      },
      {
        name: 'finalize_conversation',
        description: 'Complete the conversation and get final analysis results',
        inputSchema: {
          type: 'object',
          properties: {
            session_id: {
              type: 'string',
              description: 'ID of the conversation session',
            },
            summary_format: {
              type: 'string',
              enum: ['detailed', 'concise', 'actionable'],
              description: 'Format for the final summary',
            },
          },
          required: ['session_id'],
        },
      },
      {
        name: 'get_conversation_status',
        description: 'Check the status and progress of an ongoing conversation',
        inputSchema: {
          type: 'object',
          properties: {
            session_id: {
              type: 'string',
              description: 'ID of the conversation session',
            },
          },
          required: ['session_id'],
        },
      },
      {
        name: 'run_hypothesis_tournament',
        description: 'Run a competitive hypothesis tournament to find root causes. Multiple AI conversations test different theories in parallel, with evidence-based scoring and elimination rounds.',
        inputSchema: {
          type: 'object',
          properties: {
            claude_context: {
              type: 'object',
              properties: {
                attempted_approaches: {
                  type: 'array',
                  items: { type: 'string' },
                  description: 'What Claude Code already tried',
                },
                partial_findings: {
                  type: 'array',
                  description: 'Any findings Claude Code discovered',
                },
                stuck_description: {
                  type: 'string',
                  description: 'Description of where Claude Code got stuck',
                },
                code_scope: {
                  type: 'object',
                  properties: {
                    files: {
                      type: 'array',
                      items: { type: 'string' },
                      description: 'Files to analyze',
                    },
                    entry_points: {
                      type: 'array',
                      description: 'Specific functions/methods to start from',
                    },
                    service_names: {
                      type: 'array',
                      items: { type: 'string' },
                      description: 'Services involved in cross-system analysis',
                    },
                  },
                  required: ['files'],
                },
              },
              required: ['attempted_approaches', 'partial_findings', 'stuck_description', 'code_scope'],
            },
            issue: {
              type: 'string',
              description: 'Description of the issue to investigate',
            },
            tournament_config: {
              type: 'object',
              properties: {
                max_hypotheses: {
                  type: 'number',
                  minimum: 2,
                  maximum: 20,
                  description: 'Number of initial hypotheses to generate (default: 6)',
                },
                max_rounds: {
                  type: 'number',
                  minimum: 1,
                  maximum: 5,
                  description: 'Maximum tournament rounds (default: 3)',
                },
                parallel_sessions: {
                  type: 'number',
                  minimum: 1,
                  maximum: 10,
                  description: 'Max concurrent conversations (default: 4)',
                },
              },
            },
          },
          required: ['claude_context', 'issue'],
        },
      },
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  try {
    const { name, arguments: args } = request.params;

    // Check if API key is configured
    if (!deepReasoner) {
      throw new McpError(
        ErrorCode.InternalError,
        'GEMINI_API_KEY is not configured. Please set the GEMINI_API_KEY environment variable.',
      );
    }

    switch (name) {
      case 'escalate_analysis': {
        const parsed = EscalateAnalysisSchema.parse(args);

        // Validate and sanitize the Claude context
        const validatedContext = InputValidator.validateClaudeContext(parsed.claude_context);

        // Override with specific values from the parsed input
        const context: ClaudeCodeContext = {
          ...validatedContext,
          analysisBudgetRemaining: parsed.time_budget_seconds,
        };

        const result = await deepReasoner.escalateFromClaudeCode(
          context,
          parsed.analysis_type,
          parsed.depth_level || 3,
        );

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }

      case 'trace_execution_path': {
        const parsed = TraceExecutionPathSchema.parse(args);

        // Validate the entry point file path
        const validatedPath = InputValidator.validateFilePaths([parsed.entry_point.file])[0];
        if (!validatedPath) {
          throw new McpError(
            ErrorCode.InvalidParams,
            'Invalid entry point file path',
          );
        }

        const result = await deepReasoner.traceExecutionPath(
          { ...parsed.entry_point, file: validatedPath },
          parsed.max_depth,
          parsed.include_data_flow,
        );

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }

      case 'hypothesis_test': {
        const parsed = HypothesisTestSchema.parse(args);

        // Validate file paths
        const validatedFiles = InputValidator.validateFilePaths(parsed.code_scope.files);
        if (validatedFiles.length === 0) {
          throw new McpError(
            ErrorCode.InvalidParams,
            'No valid file paths provided',
          );
        }

        const result = await deepReasoner.testHypothesis(
          InputValidator.validateString(parsed.hypothesis, 2000),
          validatedFiles,
          InputValidator.validateString(parsed.test_approach, 1000),
        );

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }

      case 'cross_system_impact': {
        const parsed = CrossSystemImpactSchema.parse(args);

        // Validate file paths
        const validatedFiles = InputValidator.validateFilePaths(parsed.change_scope.files);
        if (validatedFiles.length === 0) {
          throw new McpError(
            ErrorCode.InvalidParams,
            'No valid file paths provided',
          );
        }

        const result = await deepReasoner.analyzeCrossSystemImpact(
          validatedFiles,
          parsed.impact_types,
        );

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }

      case 'performance_bottleneck': {
        const parsed = PerformanceBottleneckSchema.parse(args);

        // Validate the entry point file path
        const validatedPath = InputValidator.validateFilePaths([parsed.code_path.entry_point.file])[0];
        if (!validatedPath) {
          throw new McpError(
            ErrorCode.InvalidParams,
            'Invalid entry point file path',
          );
        }

        const result = await deepReasoner.analyzePerformance(
          { ...parsed.code_path.entry_point, file: validatedPath },
          parsed.profile_depth,
          parsed.code_path.suspected_issues ?
            InputValidator.validateStringArray(parsed.code_path.suspected_issues) :
            undefined,
        );

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }

      case 'start_conversation': {
        const parsed = StartConversationSchema.parse(args);

        // Validate and sanitize the Claude context
        const validatedContext = InputValidator.validateClaudeContext(parsed.claude_context);

        // Override default budget
        const context: ClaudeCodeContext = {
          ...validatedContext,
          analysisBudgetRemaining: 60,
        };

        const result = await deepReasoner.startConversation(
          context,
          parsed.analysis_type,
          parsed.initial_question,
        );

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }

      case 'continue_conversation': {
        const parsed = ContinueConversationSchema.parse(args);
        const result = await deepReasoner.continueConversation(
          parsed.session_id,
          parsed.message,
          parsed.include_code_snippets,
        );

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }

      case 'finalize_conversation': {
        const parsed = FinalizeConversationSchema.parse(args);
        const result = await deepReasoner.finalizeConversation(
          parsed.session_id,
          parsed.summary_format,
        );

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }

      case 'get_conversation_status': {
        const parsed = GetConversationStatusSchema.parse(args);
        const result = await deepReasoner.getConversationStatus(
          parsed.session_id,
        );

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }

      case 'run_hypothesis_tournament': {
        const parsed = RunHypothesisTournamentSchema.parse(args);

        // Validate and sanitize the Claude context
        const validatedContext = InputValidator.validateClaudeContext(parsed.claude_context);

        // Override with specific values from the parsed input
        const context: ClaudeCodeContext = {
          ...validatedContext,
          analysisBudgetRemaining: 300, // 5 minutes for tournament
        };

        const tournamentConfig = {
          maxHypotheses: parsed.tournament_config?.max_hypotheses ?? 6,
          maxRounds: parsed.tournament_config?.max_rounds ?? 3,
          parallelSessions: parsed.tournament_config?.parallel_sessions ?? 4,
        };

        const result = await deepReasoner.runHypothesisTournament(
          context,
          InputValidator.validateString(parsed.issue, 1000),
          tournamentConfig,
        );

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }

      default:
        throw new McpError(
          ErrorCode.MethodNotFound,
          `Unknown tool: ${name}`,
        );
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      throw new McpError(
        ErrorCode.InvalidParams,
        `Invalid parameters: ${error.errors.map(e => `${e.path.join('.')}: ${e.message}`).join(', ')}`,
      );
    }

    // Use ErrorClassifier for consistent error handling
    if (error instanceof Error) {
      const classification = ErrorClassifier.classify(error);

      switch (classification.category) {
        case 'session':
          throw new McpError(
            ErrorCode.InvalidRequest,
            classification.description,
          );

        case 'api':
          throw new McpError(
            ErrorCode.InternalError,
            classification.description,
          );

        case 'filesystem':
          throw new McpError(
            ErrorCode.InvalidRequest,
            classification.description,
          );

        default:
          console.error('Unhandled error in request handler:', error);
          throw new McpError(
            ErrorCode.InternalError,
            `Internal error: ${error.message}. Check server logs for details.`,
          );
      }
    }

    throw error;
  }
});

async function main() {
  logger.info('Starting Deep Code Reasoning MCP server...');

  const transport = new StdioServerTransport();
  logger.info('Connecting to transport...');

  await server.connect(transport);

  logger.info('Server connected successfully');
  logger.info(`GEMINI_API_KEY: ${GEMINI_API_KEY ? 'configured' : 'NOT CONFIGURED - server will return errors'}`);
  logger.info('Using Gemini model: gemini-2.5-pro-preview-05-06');
  logger.info('Ready to handle requests');
}

main().catch((error) => {
  logger.error('Fatal error in main():', error);
  process.exit(1);
});