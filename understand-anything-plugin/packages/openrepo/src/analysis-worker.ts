import { execFile, execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  buildFileAnalysisPrompt,
  buildProjectSummaryPrompt,
  detectLayers,
  generateHeuristicTour,
  parseFileAnalysisResponse,
  parseProjectSummaryResponse,
  validateGraph,
  type EdgeType,
  type GraphEdge,
  type GraphNode,
  type KnowledgeGraph,
  type LLMFileAnalysis,
} from "@understand-anything/core";
import { createAgentClient, type OpenAICompatibleAgentClient } from "./agent-client.js";
import { OpenRepoStore } from "./store.js";
import type { OpenRepoJob, OpenRepoProject } from "./types.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pluginRoot = path.resolve(__dirname, "..", "..", "..");
const understandSkillDir = path.join(pluginRoot, "skills", "understand");
const MAX_FILE_PROMPT_CHARS = 12000;
const MAX_SAMPLE_CHARS = 4000;

interface ScanFile {
  path: string;
  language: string;
  sizeLines: number;
  fileCategory: string;
}

interface ScanOutput {
  files: ScanFile[];
  totalFiles: number;
  stats?: {
    byLanguage?: Record<string, number>;
  };
}

interface ImportMapOutput {
  importMap: Record<string, string[]>;
}

interface ExtractResult {
  path: string;
  language: string;
  fileCategory: string;
  totalLines: number;
  functions?: Array<{ name: string; startLine?: number; endLine?: number; params?: string[] }>;
  classes?: Array<{ name: string; startLine?: number; endLine?: number; methods?: string[]; properties?: string[] }>;
  callGraph?: Array<{ caller: string; callee: string; lineNumber?: number }>;
  definitions?: Array<{ name: string; kind: string; startLine?: number; endLine?: number; fields?: string[] }>;
  services?: Array<{ name: string; image?: string; startLine?: number; endLine?: number }>;
  endpoints?: Array<{ method?: string; path: string; startLine?: number; endLine?: number }>;
  steps?: Array<{ name: string; startLine?: number; endLine?: number }>;
  resources?: Array<{ name: string; kind: string; startLine?: number; endLine?: number }>;
}

interface ExtractOutput {
  results: ExtractResult[];
  filesSkipped?: string[];
}

export class OpenRepoAnalysisWorker {
  private readonly runningProjects = new Set<string>();

  constructor(private readonly store: OpenRepoStore) {}

  start(projectId: string): void {
    if (this.runningProjects.has(projectId)) return;
    this.runningProjects.add(projectId);
    void this.run(projectId).finally(() => {
      this.runningProjects.delete(projectId);
    });
  }

  private async run(projectId: string): Promise<void> {
    let job: OpenRepoJob | undefined;
    try {
      job = this.store.claimNextJob(projectId);
      const project = this.store.readProject(projectId);
      await this.analyzeProject(project, job);
      this.store.completeJob(job.id);
      this.store.appendJobLog(job.id, "Analysis completed.");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (job) {
        this.store.appendJobLog(job.id, `Analysis failed: ${message}`);
        this.store.failJob(job.id, message);
      }
    }
  }

  private async analyzeProject(project: OpenRepoProject, job: OpenRepoJob): Promise<void> {
    const settings = this.store.readSettings();
    const client = createAgentClient(settings.agent, this.store.home);
    const workDir = path.join(project.sourcePath, ".understand-anything");
    const intermediateDir = path.join(workDir, "intermediate");
    const tmpDir = path.join(workDir, "tmp");
    fs.mkdirSync(intermediateDir, { recursive: true });
    fs.mkdirSync(tmpDir, { recursive: true });

    this.log(job, "Scanning project files.", "scan", 5);
    const scan = await this.runScan(project.sourcePath, intermediateDir);
    this.log(job, `Scanned ${scan.files.length} files.`, "scan", 12);

    this.log(job, "Resolving project-internal imports.", "imports", 15);
    const importMap = await this.runImportMap(project.sourcePath, scan.files, intermediateDir);

    this.log(job, "Extracting deterministic structure.", "structure", 22);
    const extraction = await this.runStructureExtraction(project.sourcePath, scan.files, importMap, tmpDir);
    if (extraction.filesSkipped?.length) {
      this.store.appendJobLog(job.id, `Skipped ${extraction.filesSkipped.length} unreadable files.`);
    }

    this.log(job, "Generating project summary with Agent API.", "project-summary", 30);
    const projectSummary = await this.buildProjectSummary(client, project, scan.files);

    this.log(job, "Generating semantic file summaries with Agent API.", "file-analysis", 35);
    const fileAnalyses = await this.analyzeFiles(client, job, project.sourcePath, extraction.results, projectSummary.description);

    this.log(job, "Assembling knowledge graph.", "assemble", 82);
    const graph = this.assembleGraph(project, scan, importMap, extraction.results, fileAnalyses, projectSummary);
    const validation = validateGraph(graph);
    if (!validation.success || !validation.data) {
      throw new Error(validation.fatal ?? validation.errors?.[0] ?? "Generated graph did not pass validation.");
    }

    this.store.writeProjectGraph(project.id, validation.data);
    fs.writeFileSync(path.join(workDir, "meta.json"), `${JSON.stringify({
      lastAnalyzedAt: new Date().toISOString(),
      gitCommitHash: graph.project.gitCommitHash,
      version: graph.version,
      analyzedFiles: scan.files.length,
    }, null, 2)}\n`, "utf8");
    this.log(job, `Knowledge graph saved to ${project.graphPath}.`, "saving", 96);
  }

  private async runScan(projectRoot: string, intermediateDir: string): Promise<ScanOutput> {
    const outputPath = path.join(intermediateDir, "scan-result.json");
    await runNodeScript(path.join(understandSkillDir, "scan-project.mjs"), [projectRoot, outputPath]);
    return readJsonFile<ScanOutput>(outputPath);
  }

  private async runImportMap(projectRoot: string, files: ScanFile[], intermediateDir: string): Promise<Record<string, string[]>> {
    const inputPath = path.join(intermediateDir, "import-map-input.json");
    const outputPath = path.join(intermediateDir, "import-map.json");
    fs.writeFileSync(inputPath, `${JSON.stringify({ projectRoot, files }, null, 2)}\n`, "utf8");
    await runNodeScript(path.join(understandSkillDir, "extract-import-map.mjs"), [inputPath, outputPath]);
    return readJsonFile<ImportMapOutput>(outputPath).importMap ?? {};
  }

  private async runStructureExtraction(
    projectRoot: string,
    files: ScanFile[],
    importMap: Record<string, string[]>,
    tmpDir: string,
  ): Promise<ExtractOutput> {
    const inputPath = path.join(tmpDir, "openrepo-worker-extract-input.json");
    const outputPath = path.join(tmpDir, "openrepo-worker-extract-output.json");
    fs.writeFileSync(inputPath, `${JSON.stringify({
      projectRoot,
      batchFiles: files,
      batchImportData: importMap,
    }, null, 2)}\n`, "utf8");
    await runNodeScript(path.join(understandSkillDir, "extract-structure.mjs"), [inputPath, outputPath]);
    return readJsonFile<ExtractOutput>(outputPath);
  }

  private async buildProjectSummary(
    client: OpenAICompatibleAgentClient,
    project: OpenRepoProject,
    files: ScanFile[],
  ): Promise<{ description: string; frameworks: string[] }> {
    const sampleFiles = files
      .filter((file) => ["README.md", "package.json", "pyproject.toml", "Cargo.toml", "go.mod"].includes(path.basename(file.path)))
      .slice(0, 5)
      .map((file) => ({ path: file.path, content: readTextSample(path.join(project.sourcePath, file.path), MAX_SAMPLE_CHARS) }))
      .filter((file) => file.content);
    const response = await client.createChatCompletion({
      messages: [{ role: "user", content: buildProjectSummaryPrompt(files.map((file) => file.path), sampleFiles) }],
      temperature: 0.2,
      maxTokens: 1200,
    });
    const parsed = parseProjectSummaryResponse(response);
    return {
      description: parsed?.description || `${project.name} analyzed by OpenRepoCopilot.`,
      frameworks: parsed?.frameworks ?? [],
    };
  }

  private async analyzeFiles(
    client: OpenAICompatibleAgentClient,
    job: OpenRepoJob,
    projectRoot: string,
    results: ExtractResult[],
    projectContext: string,
  ): Promise<Map<string, LLMFileAnalysis>> {
    const settings = this.store.readSettings();
    const concurrency = Math.max(1, settings.agent.maxConcurrency);
    const analyses = new Map<string, LLMFileAnalysis>();
    let cursor = 0;
    let completed = 0;

    const worker = async () => {
      while (cursor < results.length) {
        const result = results[cursor++];
        const content = readTextSample(path.join(projectRoot, result.path), MAX_FILE_PROMPT_CHARS);
        try {
          const prompt = buildFileAnalysisPrompt(result.path, content, projectContext);
          const response = await client.createChatCompletion({
            messages: [{ role: "user", content: prompt }],
            temperature: 0.2,
            maxTokens: 1000,
          });
          const parsed = parseFileAnalysisResponse(response);
          analyses.set(result.path, parsed ?? fallbackFileAnalysis(result));
        } catch (error) {
          this.store.appendJobLog(job.id, `File analysis fallback for ${result.path}: ${error instanceof Error ? error.message : String(error)}`);
          analyses.set(result.path, fallbackFileAnalysis(result));
        } finally {
          completed += 1;
          const progress = 35 + Math.floor((completed / Math.max(1, results.length)) * 42);
          this.store.updateJobStatus(job.id, { phase: "file-analysis", progress });
        }
      }
    };

    await Promise.all(Array.from({ length: Math.min(concurrency, results.length || 1) }, () => worker()));
    return analyses;
  }

  private assembleGraph(
    project: OpenRepoProject,
    scan: ScanOutput,
    importMap: Record<string, string[]>,
    results: ExtractResult[],
    analyses: Map<string, LLMFileAnalysis>,
    projectSummary: { description: string; frameworks: string[] },
  ): KnowledgeGraph {
    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];
    const nodeIds = new Set<string>();
    const edgeKeys = new Set<string>();
    const fileNodeIds = new Map<string, string>();

    for (const result of results) {
      const analysis = analyses.get(result.path) ?? fallbackFileAnalysis(result);
      const fileNode = makeFileNode(result, analysis);
      addNode(nodes, nodeIds, fileNode);
      fileNodeIds.set(result.path, fileNode.id);

      for (const fn of result.functions ?? []) {
        const id = `function:${result.path}:${fn.name}`;
        addNode(nodes, nodeIds, {
          id,
          type: "function",
          name: fn.name,
          filePath: result.path,
          lineRange: lineRange(fn.startLine, fn.endLine),
          summary: analysis.functionSummaries[fn.name] ?? `Function ${fn.name}.`,
          tags: [],
          complexity: analysis.complexity,
        });
        addEdge(edges, edgeKeys, fileNode.id, id, "contains", 1);
      }

      for (const cls of result.classes ?? []) {
        const id = `class:${result.path}:${cls.name}`;
        addNode(nodes, nodeIds, {
          id,
          type: "class",
          name: cls.name,
          filePath: result.path,
          lineRange: lineRange(cls.startLine, cls.endLine),
          summary: analysis.classSummaries[cls.name] ?? `Class ${cls.name}.`,
          tags: [],
          complexity: analysis.complexity,
        });
        addEdge(edges, edgeKeys, fileNode.id, id, "contains", 1);
      }

      addNonCodeChildren(nodes, edges, nodeIds, edgeKeys, fileNode, result, analysis.complexity);

      for (const call of result.callGraph ?? []) {
        const source = `function:${result.path}:${call.caller}`;
        const target = `function:${result.path}:${call.callee}`;
        if (nodeIds.has(source) && nodeIds.has(target)) addEdge(edges, edgeKeys, source, target, "calls", 0.8);
      }
    }

    for (const [sourcePath, targets] of Object.entries(importMap)) {
      const source = fileNodeIds.get(sourcePath);
      if (!source) continue;
      for (const targetPath of targets) {
        const target = fileNodeIds.get(targetPath);
        if (target) addEdge(edges, edgeKeys, source, target, "imports", 0.7);
      }
    }

    const graph: KnowledgeGraph = {
      version: "1.0.0",
      kind: project.type === "document_kb" ? "knowledge" : "codebase",
      project: {
        name: project.name,
        languages: Object.keys(scan.stats?.byLanguage ?? {}).sort(),
        frameworks: projectSummary.frameworks,
        description: projectSummary.description,
        analyzedAt: new Date().toISOString(),
        gitCommitHash: gitCommitHash(project.sourcePath),
      },
      nodes,
      edges,
      layers: [],
      tour: [],
    };
    graph.layers = detectLayers(graph);
    graph.tour = generateHeuristicTour(graph);
    return graph;
  }

  private log(job: OpenRepoJob, message: string, phase: string, progress: number): void {
    this.store.appendJobLog(job.id, message);
    this.store.updateJobStatus(job.id, { phase, progress });
  }
}

function makeFileNode(result: ExtractResult, analysis: LLMFileAnalysis): GraphNode {
  return {
    id: `${nodeTypeForCategory(result)}:${result.path}`,
    type: nodeTypeForCategory(result),
    name: path.basename(result.path),
    filePath: result.path,
    summary: analysis.fileSummary || fallbackFileSummary(result),
    tags: analysis.tags.length ? analysis.tags : [result.language, result.fileCategory].filter(Boolean),
    complexity: analysis.complexity,
    languageNotes: analysis.languageNotes,
  };
}

function nodeTypeForCategory(result: ExtractResult): GraphNode["type"] {
  if (result.fileCategory === "config") return "config";
  if (result.fileCategory === "docs" || result.fileCategory === "markup") return "document";
  if (result.fileCategory === "infra") return "service";
  if (result.fileCategory === "data") return result.language === "sql" ? "table" : "document";
  return "file";
}

function addNonCodeChildren(
  nodes: GraphNode[],
  edges: GraphEdge[],
  nodeIds: Set<string>,
  edgeKeys: Set<string>,
  parent: GraphNode,
  result: ExtractResult,
  complexity: GraphNode["complexity"],
): void {
  for (const item of result.definitions ?? []) {
    const type = item.kind === "table" || item.kind === "view" ? "table" : "schema";
    addChild(nodes, edges, nodeIds, edgeKeys, parent.id, {
      id: `${type}:${result.path}:${item.name}`,
      type,
      name: item.name,
      filePath: result.path,
      lineRange: lineRange(item.startLine, item.endLine),
      summary: `${item.kind}: ${item.name}`,
      tags: [item.kind],
      complexity,
    });
  }
  for (const item of result.services ?? []) {
    addChild(nodes, edges, nodeIds, edgeKeys, parent.id, {
      id: `service:${result.path}:${item.name}`,
      type: "service",
      name: item.name,
      filePath: result.path,
      lineRange: lineRange(item.startLine, item.endLine),
      summary: item.image ? `Service ${item.name} using ${item.image}.` : `Service ${item.name}.`,
      tags: ["service"],
      complexity,
    });
  }
  for (const item of result.endpoints ?? []) {
    const name = `${item.method ?? ""} ${item.path}`.trim();
    addChild(nodes, edges, nodeIds, edgeKeys, parent.id, {
      id: `endpoint:${result.path}:${name}`,
      type: "endpoint",
      name,
      filePath: result.path,
      lineRange: lineRange(item.startLine, item.endLine),
      summary: `Endpoint ${name}.`,
      tags: ["endpoint"],
      complexity,
    });
  }
  for (const item of result.steps ?? []) {
    addChild(nodes, edges, nodeIds, edgeKeys, parent.id, {
      id: `pipeline:${result.path}:${item.name}`,
      type: "pipeline",
      name: item.name,
      filePath: result.path,
      lineRange: lineRange(item.startLine, item.endLine),
      summary: `Pipeline step ${item.name}.`,
      tags: ["pipeline"],
      complexity,
    });
  }
  for (const item of result.resources ?? []) {
    addChild(nodes, edges, nodeIds, edgeKeys, parent.id, {
      id: `resource:${result.path}:${item.name}`,
      type: "resource",
      name: item.name,
      filePath: result.path,
      lineRange: lineRange(item.startLine, item.endLine),
      summary: `${item.kind} resource ${item.name}.`,
      tags: [item.kind],
      complexity,
    });
  }
}

function addChild(nodes: GraphNode[], edges: GraphEdge[], nodeIds: Set<string>, edgeKeys: Set<string>, parentId: string, node: GraphNode): void {
  addNode(nodes, nodeIds, node);
  addEdge(edges, edgeKeys, parentId, node.id, "contains", 1);
}

function addNode(nodes: GraphNode[], nodeIds: Set<string>, node: GraphNode): void {
  if (nodeIds.has(node.id)) return;
  nodeIds.add(node.id);
  nodes.push(node);
}

function addEdge(edges: GraphEdge[], edgeKeys: Set<string>, source: string, target: string, type: EdgeType, weight: number): void {
  const key = `${type}|${source}|${target}`;
  if (edgeKeys.has(key)) return;
  edgeKeys.add(key);
  edges.push({ source, target, type, direction: "forward", weight });
}

function fallbackFileAnalysis(result: ExtractResult): LLMFileAnalysis {
  return {
    fileSummary: fallbackFileSummary(result),
    tags: [result.language, result.fileCategory].filter(Boolean),
    complexity: result.totalLines > 300 ? "complex" : result.totalLines > 80 ? "moderate" : "simple",
    functionSummaries: Object.fromEntries((result.functions ?? []).map((fn) => [fn.name, `Function ${fn.name}.`])),
    classSummaries: Object.fromEntries((result.classes ?? []).map((cls) => [cls.name, `Class ${cls.name}.`])),
  };
}

function fallbackFileSummary(result: ExtractResult): string {
  const parts = [`${result.path} is a ${result.fileCategory} file`];
  if (result.functions?.length) parts.push(`with ${result.functions.length} functions`);
  if (result.classes?.length) parts.push(`with ${result.classes.length} classes`);
  return `${parts.join(" ")}.`;
}

function lineRange(start: number | undefined, end: number | undefined): [number, number] | undefined {
  return typeof start === "number" && typeof end === "number" ? [start, end] : undefined;
}

function readTextSample(filePath: string, maxChars: number): string {
  try {
    const buffer = fs.readFileSync(filePath);
    if (buffer.includes(0)) return "";
    return buffer.toString("utf8").slice(0, maxChars);
  } catch {
    return "";
  }
}

function readJsonFile<T>(filePath: string): T {
  return JSON.parse(fs.readFileSync(filePath, "utf8")) as T;
}

function runNodeScript(scriptPath: string, args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    execFile(process.execPath, [scriptPath, ...args], { cwd: pluginRoot, maxBuffer: 20 * 1024 * 1024 }, (error, stdout, stderr) => {
      void stdout;
      if (error) {
        reject(new Error(`${path.basename(scriptPath)} failed: ${stderr || error.message}`));
      } else {
        resolve();
      }
    });
  });
}

function gitCommitHash(projectRoot: string): string {
  try {
    return execFileSync("git", ["-C", projectRoot, "rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  } catch {
    return "";
  }
}
