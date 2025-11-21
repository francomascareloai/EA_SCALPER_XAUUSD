# BMM Workflows

## Available Workflows in bmm

**brainstorm-project**
- Path: `.bmad/bmm/workflows/1-analysis/brainstorm-project/workflow.yaml`
- Facilitate project brainstorming sessions by orchestrating the CIS brainstorming workflow with project-specific context and guidance.

**domain-research**
- Path: `.bmad/bmm/workflows/1-analysis/domain-research/workflow.yaml`
- Collaborative exploration of domain-specific requirements, regulations, and patterns for complex projects

**product-brief**
- Path: `.bmad/bmm/workflows/1-analysis/product-brief/workflow.yaml`
- Interactive product brief creation workflow that guides users through defining their product vision with multiple input sources and conversational collaboration

**research**
- Path: `.bmad/bmm/workflows/1-analysis/research/workflow.yaml`
- Adaptive research workflow supporting multiple research types: market research, deep research prompt generation, technical/architecture evaluation, competitive intelligence, user research, and domain analysis

**create-ux-design**
- Path: `.bmad/bmm/workflows/2-plan-workflows/create-ux-design/workflow.yaml`
- Collaborative UX design facilitation workflow that creates exceptional user experiences through visual exploration and informed decision-making. Unlike template-driven approaches, this workflow facilitates discovery, generates visual options, and collaboratively designs the UX with the user at every step.

**prd**
- Path: `.bmad/bmm/workflows/2-plan-workflows/prd/workflow.yaml`
- Unified PRD workflow for BMad Method and Enterprise Method tracks. Produces strategic PRD and tactical epic breakdown. Hands off to architecture workflow for technical design. Note: Quick Flow track uses tech-spec workflow.

**tech-spec**
- Path: `.bmad/bmm/workflows/2-plan-workflows/tech-spec/workflow.yaml`
- Technical specification workflow for quick-flow projects. Creates focused tech spec and generates epic + stories (1 story for simple changes, 2-5 stories for features). Tech-spec only - no PRD needed.

**architecture**
- Path: `.bmad/bmm/workflows/3-solutioning/architecture/workflow.yaml`
- Collaborative architectural decision facilitation for AI-agent consistency. Replaces template-driven architecture with intelligent, adaptive conversation that produces a decision-focused architecture document optimized for preventing agent conflicts.

**create-epics-and-stories**
- Path: `.bmad/bmm/workflows/3-solutioning/create-epics-and-stories/workflow.yaml`
- Transform PRD requirements into bite-sized stories organized into deliverable functional epics. This workflow takes a Product Requirements Document (PRD) and breaks it down into epics and user stories that can be easily assigned to development teams. It ensures that all functional requirements are captured in a structured format, making it easier for teams to understand and implement the necessary features.

**implementation-readiness**
- Path: `.bmad/bmm/workflows/3-solutioning/implementation-readiness/workflow.yaml`
- Validate that PRD, UX Design, Architecture, Epics and Stories are complete and aligned before Phase 4 implementation. Ensures all artifacts cover the MVP requirements with no gaps or contradictions.

**code-review**
- Path: `.bmad/bmm/workflows/4-implementation/code-review/workflow.yaml`
- Perform a Senior Developer code review on a completed story flagged Ready for Review, leveraging story-context, epic tech-spec, repo docs, MCP servers for latest best-practices, and web search as fallback. Appends structured review notes to the story.

**correct-course**
- Path: `.bmad/bmm/workflows/4-implementation/correct-course/workflow.yaml`
- Navigate significant changes during sprint execution by analyzing impact, proposing solutions, and routing for implementation

**create-story**
- Path: `.bmad/bmm/workflows/4-implementation/create-story/workflow.yaml`
- Create the next user story markdown from epics/PRD and architecture, using a standard template and saving to the stories folder

**dev-story**
- Path: `.bmad/bmm/workflows/4-implementation/dev-story/workflow.yaml`
- Execute a story by implementing tasks/subtasks, writing tests, validating, and updating the story file per acceptance criteria

**epic-tech-context**
- Path: `.bmad/bmm/workflows/4-implementation/epic-tech-context/workflow.yaml`
- Generate a comprehensive Technical Specification from PRD and Architecture with acceptance criteria and traceability mapping

**retrospective**
- Path: `.bmad/bmm/workflows/4-implementation/retrospective/workflow.yaml`
- Run after epic completion to review overall success, extract lessons learned, and explore if new information emerged that might impact the next epic

**sprint-planning**
- Path: `.bmad/bmm/workflows/4-implementation/sprint-planning/workflow.yaml`
- Generate and manage the sprint status tracking file for Phase 4 implementation, extracting all epics and stories from epic files and tracking their status through the development lifecycle

**story-context**
- Path: `.bmad/bmm/workflows/4-implementation/story-context/workflow.yaml`
- Assemble a dynamic Story Context XML by pulling latest documentation and existing code/library artifacts relevant to a drafted story

**story-done**
- Path: `.bmad/bmm/workflows/4-implementation/story-done/workflow.yaml`
- Marks a story as done (DoD complete) and moves it from its current status → DONE in the status file. Advances the story queue. Simple status-update workflow with no searching required.

**story-ready**
- Path: `.bmad/bmm/workflows/4-implementation/story-ready/workflow.yaml`
- Marks a drafted story as ready for development and moves it from TODO → IN PROGRESS in the status file. Simple status-update workflow with no searching required.

**create-excalidraw-dataflow**
- Path: `.bmad/bmm/workflows/diagrams/create-dataflow/workflow.yaml`
- Create data flow diagrams (DFD) in Excalidraw format

**create-excalidraw-diagram**
- Path: `.bmad/bmm/workflows/diagrams/create-diagram/workflow.yaml`
- Create system architecture diagrams, ERDs, UML diagrams, or general technical diagrams in Excalidraw format

**create-excalidraw-flowchart**
- Path: `.bmad/bmm/workflows/diagrams/create-flowchart/workflow.yaml`
- Create a flowchart visualization in Excalidraw format for processes, pipelines, or logic flows

**create-excalidraw-wireframe**
- Path: `.bmad/bmm/workflows/diagrams/create-wireframe/workflow.yaml`
- Create website or app wireframes in Excalidraw format

**document-project**
- Path: `.bmad/bmm/workflows/document-project/workflow.yaml`
- Analyzes and documents brownfield projects by scanning codebase, architecture, and patterns to create comprehensive reference documentation for AI-assisted development

**workflow-init**
- Path: `.bmad/bmm/workflows/workflow-status/init/workflow.yaml`
- Initialize a new BMM project by determining level, type, and creating workflow path

**workflow-status**
- Path: `.bmad/bmm/workflows/workflow-status/workflow.yaml`
- Lightweight status checker - answers "what should I do now?" for any agent. Reads YAML status file for workflow tracking. Use workflow-init for new projects.


## Execution

When running any workflow:
1. LOAD {project-root}/.bmad/core/tasks/workflow.xml
2. Pass the workflow path as 'workflow-config' parameter
3. Follow workflow.xml instructions EXACTLY
4. Save outputs after EACH section

## Modes
- Normal: Full interaction
- #yolo: Skip optional steps
