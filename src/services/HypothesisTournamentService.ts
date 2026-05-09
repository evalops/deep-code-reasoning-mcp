import type {
  ClaudeCodeContext,
  HypothesisDefinition,
  HypothesisExplorationResult,
  TournamentResult,
  TournamentRound,
  Evidence,
  Finding,
  Action,
} from '../models/types.js';
import { ConversationalGeminiService } from './ConversationalGeminiService.js';
import { ConversationManager } from './ConversationManager.js';
import { SecureCodeReader } from '../utils/SecureCodeReader.js';

interface TournamentConfig {
  maxHypotheses: number;
  maxRounds: number;
  eliminationThreshold: number; // Confidence below this gets eliminated
  parallelSessions: number; // Max concurrent conversations
  crossPollinationEnabled: boolean;
}

export class HypothesisTournamentService {
  private conversationalGemini: ConversationalGeminiService;
  private conversationManager: ConversationManager;
  private codeReader: SecureCodeReader;
  private config: TournamentConfig;

  constructor(
    geminiApiKey: string,
    config: Partial<TournamentConfig> = {},
  ) {
    this.conversationalGemini = new ConversationalGeminiService(geminiApiKey);
    this.conversationManager = new ConversationManager();
    this.codeReader = new SecureCodeReader();
    this.config = {
      maxHypotheses: 6,
      maxRounds: 3,
      eliminationThreshold: 0.3,
      parallelSessions: 4,
      crossPollinationEnabled: true,
      ...config,
    };
  }

  /**
   * Run a hypothesis tournament to find the root cause of an issue
   */
  async runTournament(
    context: ClaudeCodeContext,
    issue: string,
  ): Promise<TournamentResult> {
    const startTime = Date.now();
    
    // Generate initial hypotheses
    const hypotheses = await this.generateHypotheses(context, issue);
    
    const rounds: TournamentRound[] = [];
    let remainingHypotheses = [...hypotheses];
    let allFindings: Finding[] = [];

    // Run tournament rounds
    for (let roundNum = 1; roundNum <= this.config.maxRounds && remainingHypotheses.length > 1; roundNum++) {
      const round = await this.runRound(
        roundNum,
        remainingHypotheses,
        context,
        issue,
        rounds,
      );
      
      rounds.push(round);
      allFindings.push(...this.extractFindingsFromRound(round));
      
      // Eliminate low-confidence hypotheses
      remainingHypotheses = round.results
        .filter(r => r.overallConfidence >= this.config.eliminationThreshold)
        .sort((a, b) => b.overallConfidence - a.overallConfidence)
        .slice(0, Math.ceil(remainingHypotheses.length / 2))
        .map(r => r.hypothesis);
      
      // Share insights across sessions if enabled
      if (this.config.crossPollinationEnabled && remainingHypotheses.length > 1) {
        await this.crossPollinateInsights(round.results);
      }
    }

    // Determine winner and runner-up
    const finalResults = rounds[rounds.length - 1]?.results || [];
    const sortedResults = finalResults.sort((a, b) => b.overallConfidence - a.overallConfidence);
    
    const winner = sortedResults[0];
    const runnerUp = sortedResults[1];

    // Calculate metrics
    const duration = Date.now() - startTime;
    const sequentialTime = hypotheses.length * (duration / rounds.length);
    const parallelEfficiency = sequentialTime / duration;

    return {
      issue,
      totalHypotheses: hypotheses.length,
      rounds,
      winner,
      runnerUp,
      allFindings,
      recommendations: this.generateRecommendations(winner, runnerUp, allFindings),
      duration,
      parallelEfficiency,
    };
  }

  /**
   * Generate initial hypotheses based on the issue description
   */
  private async generateHypotheses(
    context: ClaudeCodeContext,
    issue: string,
  ): Promise<HypothesisDefinition[]> {
    // Read relevant code files
    const codeFiles = await this.codeReader.readCodeFiles(context.focusArea);
    
    // Use Gemini to generate hypotheses
    const sessionId = this.conversationManager.createSession(context);
    
    const prompt = `Given this issue: "${issue}"

And considering:
- Previous attempts: ${context.attemptedApproaches.join(', ')}
- Partial findings: ${context.partialFindings.map(f => f.description).join(', ')}

Generate ${this.config.maxHypotheses} distinct hypotheses for what might be causing this issue. For each hypothesis, provide:
1. A clear theory about the root cause
2. A specific approach to test it
3. A category (performance/bug/security/architecture/integration)
4. A priority score (0-1) based on likelihood

Focus on diverse hypotheses that cover different aspects of the system.`;

    const { response } = await this.conversationalGemini.startConversation(
      sessionId,
      context,
      'hypothesis_generation',
      codeFiles,
      prompt,
    );

    // Parse hypotheses from response
    const hypotheses = this.parseHypothesesFromResponse(response);
    
    // Clean up generation session
    this.conversationManager.releaseLock(sessionId);
    
    return hypotheses.slice(0, this.config.maxHypotheses);
  }

  /**
   * Run a single tournament round
   */
  private async runRound(
    roundNumber: number,
    hypotheses: HypothesisDefinition[],
    context: ClaudeCodeContext,
    issue: string,
    previousRounds: TournamentRound[],
  ): Promise<TournamentRound> {
    const roundStartTime = Date.now();
    
    // Create sessions for each hypothesis
    const sessions = hypotheses.map(h => ({
      hypothesis: h,
      sessionId: this.conversationManager.createSession({
        ...context,
        stuckPoints: [...context.stuckPoints, `Testing: ${h.theory}`],
      }),
    }));

    // Read code files once for all sessions
    const codeFiles = await this.codeReader.readCodeFiles(context.focusArea);

    // Explore hypotheses in parallel (respecting parallelism limit)
    const results: HypothesisExplorationResult[] = [];
    
    for (let i = 0; i < sessions.length; i += this.config.parallelSessions) {
      const batch = sessions.slice(i, i + this.config.parallelSessions);
      const batchResults = await Promise.all(
        batch.map(({ hypothesis, sessionId }) =>
          this.exploreHypothesis(
            sessionId,
            hypothesis,
            issue,
            codeFiles,
            roundNumber,
            previousRounds,
          ),
        ),
      );
      results.push(...batchResults);
    }

    // Extract cross-hypothesis insights
    const insights = this.extractCrossHypothesisInsights(results);

    return {
      roundNumber,
      hypotheses,
      results,
      eliminatedHypotheses: hypotheses
        .filter(h => !results.find(r => r.hypothesis.id === h.id && r.overallConfidence >= this.config.eliminationThreshold))
        .map(h => h.id),
      insights,
    };
  }

  /**
   * Explore a single hypothesis through conversational analysis
   */
  private async exploreHypothesis(
    sessionId: string,
    hypothesis: HypothesisDefinition,
    issue: string,
    codeFiles: Map<string, string>,
    roundNumber: number,
    previousRounds: TournamentRound[],
  ): Promise<HypothesisExplorationResult> {
    const evidence: Evidence[] = [];
    const keyInsights: string[] = [];
    let explorationDepth = 0;

    try {
      // Initial prompt based on round
      const initialPrompt = this.buildExplorationPrompt(
        hypothesis,
        issue,
        roundNumber,
        previousRounds,
      );

      // Start the exploration
      const { response: initialResponse, suggestedFollowUps } = 
        await this.conversationalGemini.startConversation(
          sessionId,
          this.conversationManager.getSession(sessionId)!.context,
          'hypothesis_test',
          codeFiles,
          initialPrompt,
        );

      explorationDepth++;
      
      // Extract initial evidence
      evidence.push(...this.extractEvidenceFromResponse(initialResponse, 'initial'));
      keyInsights.push(...this.extractInsightsFromResponse(initialResponse));

      // Follow up based on initial findings
      if (suggestedFollowUps.length > 0 && roundNumber > 1) {
        const followUpPrompt = `Based on your initial analysis, please investigate: ${suggestedFollowUps[0]}
Focus on finding concrete evidence that either supports or contradicts the hypothesis.`;

        const { response: followUpResponse } = await this.conversationalGemini.continueConversation(
          sessionId,
          followUpPrompt,
          true,
        );

        explorationDepth++;
        evidence.push(...this.extractEvidenceFromResponse(followUpResponse, 'followup'));
      }

      // Try to reproduce the issue if we have enough confidence
      const currentConfidence = this.calculateConfidence(evidence);
      if (currentConfidence > 0.5) {
        const reproductionResponse = await this.attemptReproduction(
          sessionId,
          hypothesis,
          evidence,
        );
        
        explorationDepth++;
        if (reproductionResponse.success) {
          evidence.push({
            type: 'supporting',
            description: 'Successfully reproduced the issue',
            confidence: 0.9,
            discoveredAt: Date.now(),
          });
        }
      }

      // Finalize and get structured results
      const finalResult = await this.conversationalGemini.finalizeConversation(
        sessionId,
        'actionable',
      );

      const relatedFindings = finalResult.findings.rootCauses
        .filter(rc => rc.confidence < 0.5) // Lower confidence findings might be unrelated
        .map(rc => ({
          type: 'bug' as const,
          severity: 'medium' as const,
          location: rc.evidence[0] || { file: 'unknown', line: 0 },
          description: rc.description,
          evidence: [rc.description],
        }));

      return {
        hypothesis,
        sessionId,
        evidence,
        overallConfidence: this.calculateConfidence(evidence),
        explorationDepth,
        keyInsights,
        relatedFindings: relatedFindings.length > 0 ? relatedFindings : undefined,
      };
    } catch (error) {
      console.error(`Error exploring hypothesis ${hypothesis.id}:`, error);
      
      // Return low-confidence result on error
      return {
        hypothesis,
        sessionId,
        evidence: [{
          type: 'contradicting',
          description: `Exploration failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          confidence: 0.1,
          discoveredAt: Date.now(),
        }],
        overallConfidence: 0.1,
        explorationDepth,
        keyInsights: ['Exploration encountered errors'],
      };
    }
  }

  /**
   * Build exploration prompt based on round and previous findings
   */
  private buildExplorationPrompt(
    hypothesis: HypothesisDefinition,
    issue: string,
    roundNumber: number,
    previousRounds: TournamentRound[],
  ): string {
    let prompt = `We're investigating this issue: "${issue}"

Current hypothesis: "${hypothesis.theory}"
Test approach: ${hypothesis.testApproach}
Category: ${hypothesis.category}

Please explore this hypothesis by:
1. Looking for evidence that supports or contradicts it
2. Examining the relevant code sections
3. Considering edge cases and boundary conditions
4. Checking for patterns that match the symptoms`;

    // Add context from previous rounds
    if (roundNumber > 1 && previousRounds.length > 0) {
      const eliminatedTheories = previousRounds
        .flatMap(r => r.eliminatedHypotheses)
        .map(id => previousRounds.flatMap(r => r.hypotheses).find(h => h.id === id)?.theory)
        .filter(Boolean);

      prompt += `\n\nPreviously eliminated theories:\n${eliminatedTheories.join('\n- ')}`;
      
      const previousInsights = previousRounds.flatMap(r => r.insights);
      if (previousInsights.length > 0) {
        prompt += `\n\nInsights from previous rounds:\n${previousInsights.join('\n- ')}`;
      }
    }

    return prompt;
  }

  /**
   * Extract evidence from AI response
   */
  private extractEvidenceFromResponse(response: string, phase: string): Evidence[] {
    const evidence: Evidence[] = [];
    const timestamp = Date.now();

    // Look for supporting evidence patterns
    const supportingPatterns = [
      /confirm|validate|support|consistent with|aligns with|indicates/i,
      /found|discovered|identified|observed/i,
    ];

    // Look for contradicting evidence patterns
    const contradictingPatterns = [
      /contradict|disprove|inconsistent|rules out|unlikely/i,
      /no evidence|not found|absence of/i,
    ];

    const lines = response.split('\n');
    
    for (const line of lines) {
      const isSupporting = supportingPatterns.some(p => p.test(line));
      const isContradicting = contradictingPatterns.some(p => p.test(line));
      
      if (isSupporting || isContradicting) {
        // Extract code references if present
        const codeRef = line.match(/(\w+\.\w+):(\d+)/);
        
        evidence.push({
          type: isSupporting ? 'supporting' : 'contradicting',
          description: line.trim(),
          location: codeRef ? { file: codeRef[1], line: parseInt(codeRef[2]) } : undefined,
          confidence: this.estimateConfidenceFromText(line),
          discoveredAt: timestamp,
        });
      }
    }

    return evidence;
  }

  /**
   * Extract key insights from response
   */
  private extractInsightsFromResponse(response: string): string[] {
    const insights: string[] = [];
    
    // Look for insight patterns
    const insightPatterns = [
      /key finding:|important:|notable:|significant:/i,
      /this suggests|this indicates|this means/i,
      /pattern:|observation:/i,
    ];

    const lines = response.split('\n');
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (insightPatterns.some(p => p.test(line))) {
        // Get this line and potentially the next one
        let insight = line.trim();
        if (i + 1 < lines.length && lines[i + 1].trim() && !insightPatterns.some(p => p.test(lines[i + 1]))) {
          insight += ' ' + lines[i + 1].trim();
        }
        insights.push(insight);
      }
    }

    return insights;
  }

  /**
   * Attempt to reproduce the issue based on current evidence
   */
  private async attemptReproduction(
    sessionId: string,
    hypothesis: HypothesisDefinition,
    currentEvidence: Evidence[],
  ): Promise<{ success: boolean; steps?: string[] }> {
    const prompt = `Based on the evidence we've gathered for the hypothesis "${hypothesis.theory}", 
can you provide specific steps to reproduce this issue? 

Current evidence:
${currentEvidence.filter(e => e.type === 'supporting').map(e => `- ${e.description}`).join('\n')}

Please provide concrete reproduction steps if possible, or explain why reproduction isn't feasible.`;

    const { response } = await this.conversationalGemini.continueConversation(
      sessionId,
      prompt,
      false,
    );

    // Simple check for reproduction success
    const successPatterns = /can be reproduced|reproduction steps:|to reproduce:|reproducible/i;
    const failurePatterns = /cannot reproduce|unable to reproduce|not reproducible/i;

    if (successPatterns.test(response)) {
      // Extract steps
      const steps = response
        .split('\n')
        .filter(line => /^\d+\.|^-|^step/i.test(line.trim()))
        .map(line => line.trim());

      return { success: true, steps };
    }

    return { success: false };
  }

  /**
   * Calculate overall confidence from evidence
   */
  private calculateConfidence(evidence: Evidence[]): number {
    if (evidence.length === 0) return 0;

    const weights = { supporting: 1, contradicting: -1, neutral: 0 };
    
    const weightedSum = evidence.reduce(
      (sum, e) => sum + (weights[e.type] * e.confidence),
      0,
    );

    const maxPossible = evidence.reduce(
      (sum, e) => sum + Math.abs(weights[e.type] * e.confidence),
      0,
    );

    // Normalize to 0-1 range
    return maxPossible > 0 ? (weightedSum + maxPossible) / (2 * maxPossible) : 0.5;
  }

  /**
   * Estimate confidence from text content
   */
  private estimateConfidenceFromText(text: string): number {
    const highConfidenceWords = /definitely|certainly|clearly|obviously|confirmed/i;
    const mediumConfidenceWords = /likely|probably|suggests|indicates|appears/i;
    const lowConfidenceWords = /possibly|might|could|maybe|uncertain/i;

    if (highConfidenceWords.test(text)) return 0.8 + Math.random() * 0.2;
    if (mediumConfidenceWords.test(text)) return 0.5 + Math.random() * 0.3;
    if (lowConfidenceWords.test(text)) return 0.2 + Math.random() * 0.3;
    
    return 0.5; // Default medium confidence
  }

  /**
   * Parse hypotheses from Gemini's response
   */
  private parseHypothesesFromResponse(response: string): HypothesisDefinition[] {
    const hypotheses: HypothesisDefinition[] = [];
    
    // Look for numbered hypotheses
    const hypothesisBlocks = response.split(/\d+\.\s+/);
    
    for (let i = 1; i < hypothesisBlocks.length; i++) {
      const block = hypothesisBlocks[i];
      
      // Extract theory (usually first line)
      const lines = block.split('\n').filter(l => l.trim());
      if (lines.length === 0) continue;
      
      const theory = lines[0].replace(/theory:|hypothesis:/i, '').trim();
      
      // Extract other fields
      const testApproach = this.extractField(block, /approach:|test:|method:/i) || 
                          'Investigate through code analysis';
      const category = this.extractCategory(block);
      const priority = this.extractPriority(block);
      
      hypotheses.push({
        id: `h${i}`,
        theory,
        testApproach,
        category,
        priority,
      });
    }

    // If no numbered format found, try other patterns
    if (hypotheses.length === 0) {
      // Fallback parsing logic
      const lines = response.split('\n');
      let currentHypothesis: Partial<HypothesisDefinition> | null = null;
      
      for (const line of lines) {
        if (/hypothesis|theory/i.test(line) && !currentHypothesis) {
          currentHypothesis = {
            id: `h${hypotheses.length + 1}`,
            theory: line.replace(/.*?:/, '').trim(),
            category: 'bug',
            priority: 0.5,
          };
        } else if (currentHypothesis && /approach|test|method/i.test(line)) {
          currentHypothesis.testApproach = line.replace(/.*?:/, '').trim();
          hypotheses.push(currentHypothesis as HypothesisDefinition);
          currentHypothesis = null;
        }
      }
    }

    return hypotheses;
  }

  private extractField(text: string, pattern: RegExp): string | undefined {
    const match = text.match(new RegExp(`${pattern.source}\\s*(.+)`, 'i'));
    return match?.[1]?.trim();
  }

  private extractCategory(text: string): HypothesisDefinition['category'] {
    if (/performance|slow|latency|speed/i.test(text)) return 'performance';
    if (/security|vulnerability|exploit|injection/i.test(text)) return 'security';
    if (/architecture|design|structure|pattern/i.test(text)) return 'architecture';
    if (/integration|external|api|service/i.test(text)) return 'integration';
    return 'bug';
  }

  private extractPriority(text: string): number {
    const match = text.match(/priority:?\s*([\d.]+)|(\d+)%|likelihood:?\s*([\d.]+)/i);
    if (match) {
      const value = parseFloat(match[1] || match[2] || match[3]);
      return value > 1 ? value / 100 : value;
    }
    
    // Estimate from confidence words
    if (/high|likely|probable/i.test(text)) return 0.7 + Math.random() * 0.2;
    if (/medium|possible|moderate/i.test(text)) return 0.4 + Math.random() * 0.2;
    if (/low|unlikely|improbable/i.test(text)) return 0.1 + Math.random() * 0.2;
    
    return 0.5;
  }

  /**
   * Share insights between active sessions
   */
  private async crossPollinateInsights(results: HypothesisExplorationResult[]): Promise<void> {
    // Find significant insights that could help other hypotheses
    const significantInsights = results
      .filter(r => r.overallConfidence > 0.6)
      .flatMap(r => r.keyInsights)
      .filter(insight => 
        /pattern|common|related|connected|affects all|system-wide/i.test(insight),
      );

    if (significantInsights.length === 0) return;

    // Share with lower-confidence hypotheses
    const strugglingHypotheses = results.filter(r => r.overallConfidence < 0.5);
    
    for (const result of strugglingHypotheses) {
      try {
        const prompt = `New insights from parallel investigations:
${significantInsights.join('\n- ')}

Do any of these insights change your analysis of the hypothesis "${result.hypothesis.theory}"?`;

        await this.conversationalGemini.continueConversation(
          result.sessionId,
          prompt,
          false,
        );
      } catch (error) {
        console.warn(`Failed to cross-pollinate to session ${result.sessionId}:`, error);
      }
    }
  }

  /**
   * Extract findings from a tournament round
   */
  private extractFindingsFromRound(round: TournamentRound): Finding[] {
    return round.results
      .flatMap(r => r.relatedFindings || [])
      .filter((f, index, self) => 
        // Deduplicate findings
        index === self.findIndex(other => 
          other.description === f.description && 
          other.location.file === f.location.file,
        ),
      );
  }

  /**
   * Extract insights that span multiple hypotheses
   */
  private extractCrossHypothesisInsights(results: HypothesisExplorationResult[]): string[] {
    const insights: string[] = [];
    
    // Find common patterns
    const allInsights = results.flatMap(r => r.keyInsights);
    const insightCounts = new Map<string, number>();
    
    for (const insight of allInsights) {
      // Normalize for comparison
      const normalized = insight.toLowerCase().replace(/[^\w\s]/g, '');
      insightCounts.set(normalized, (insightCounts.get(normalized) || 0) + 1);
    }
    
    // Insights that appear in multiple hypotheses
    for (const [insight, count] of insightCounts) {
      if (count >= 2) {
        const original = allInsights.find(i => 
          i.toLowerCase().replace(/[^\w\s]/g, '') === insight,
        );
        if (original) {
          insights.push(`Common finding across ${count} hypotheses: ${original}`);
        }
      }
    }
    
    // Contradictory findings
    const highConfidence = results.filter(r => r.overallConfidence > 0.7);
    const lowConfidence = results.filter(r => r.overallConfidence < 0.3);
    
    if (highConfidence.length > 0 && lowConfidence.length > 0) {
      insights.push(
        `Strong evidence for: ${highConfidence.map(r => r.hypothesis.theory).join(', ')}. ` +
        `Weak evidence for: ${lowConfidence.map(r => r.hypothesis.theory).join(', ')}.`,
      );
    }
    
    return insights;
  }

  /**
   * Generate recommendations from tournament results
   */
  private generateRecommendations(
    winner: HypothesisExplorationResult | undefined,
    runnerUp: HypothesisExplorationResult | undefined,
    allFindings: Finding[],
  ): TournamentResult['recommendations'] {
    const primary: Action[] = [];
    const secondary: Action[] = [];

    if (winner && winner.overallConfidence > 0.7) {
      // High confidence winner
      primary.push({
        type: 'fix',
        description: `Address root cause: ${winner.hypothesis.theory}`,
        priority: 'critical',
        estimatedEffort: '2-4 hours',
      });

      // Add reproduction steps if available
      if (winner.reproductionSteps) {
        primary.push({
          type: 'investigate',
          description: 'Verify issue using reproduction steps',
          priority: 'high',
          estimatedEffort: '30 minutes',
        });
      }
    } else if (winner) {
      // Lower confidence winner
      primary.push({
        type: 'investigate',
        description: `Further investigate: ${winner.hypothesis.theory}`,
        priority: 'high',
        estimatedEffort: '1-2 hours',
      });
    }

    if (runnerUp && runnerUp.overallConfidence > 0.5) {
      secondary.push({
        type: 'investigate',
        description: `Also consider: ${runnerUp.hypothesis.theory}`,
        priority: 'medium',
        estimatedEffort: '1-2 hours',
      });
    }

    // Add recommendations for serendipitous findings
    const criticalFindings = allFindings.filter(f => f.severity === 'critical' || f.severity === 'high');
    for (const finding of criticalFindings) {
      secondary.push({
        type: 'fix',
        description: `Unrelated issue found: ${finding.description}`,
        priority: finding.severity === 'critical' ? 'high' : 'medium',
        estimatedEffort: '1-3 hours',
      });
    }

    // Add monitoring if performance-related
    if (winner?.hypothesis.category === 'performance') {
      primary.push({
        type: 'monitor',
        description: 'Set up performance monitoring for affected areas',
        priority: 'medium',
        estimatedEffort: '1 hour',
      });
    }

    return { primary, secondary };
  }
}
