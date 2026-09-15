export type WorkflowStage = 1 | 2 | 3 | 4;

interface GuidedWorkflowOptions {
  canEnter: (stage: WorkflowStage) => string | null;
  onBlocked: (message: string) => void;
  onStageChange?: (stage: WorkflowStage) => void;
}

export class GuidedWorkflow {
  private readonly stages: HTMLElement[];
  private readonly stepButtons: HTMLButtonElement[];
  private readonly completed = new Set<WorkflowStage>();
  private currentStage: WorkflowStage = 1;

  constructor(private readonly options: GuidedWorkflowOptions) {
    this.stages = Array.from(document.querySelectorAll<HTMLElement>('[data-workflow-stage]'));
    this.stepButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('button[data-workflow-step]'));
    if (this.stages.length !== 4 || this.stepButtons.length !== 4) {
      throw new Error('The guided workflow requires exactly four stages and four step buttons.');
    }
    for (const button of this.stepButtons) {
      button.addEventListener('click', () => {
        const stage = Number(button.dataset.workflowStep) as WorkflowStage;
        this.requestStage(stage);
      });
    }
    document.querySelectorAll<HTMLButtonElement>('[data-workflow-next]').forEach((button) => {
      button.addEventListener('click', () => {
        const stage = Number(button.dataset.workflowNext) as WorkflowStage;
        this.requestStage(stage);
      });
    });
    this.bindPhotoSourceTabs();
    this.activate(1, false);
  }

  get activeStage(): WorkflowStage {
    return this.currentStage;
  }

  requestStage(stage: WorkflowStage): boolean {
    if (stage > this.currentStage) {
      const problem = this.options.canEnter(stage);
      if (problem) {
        this.options.onBlocked(problem);
        return false;
      }
      this.completed.add(this.currentStage);
    }
    this.activate(stage);
    return true;
  }

  activate(stage: WorkflowStage, focus = true): void {
    this.currentStage = stage;
    for (const panel of this.stages) {
      const isActive = Number(panel.dataset.workflowStage) === stage;
      panel.hidden = !isActive;
      panel.classList.toggle('is-active', isActive);
    }
    for (const button of this.stepButtons) {
      const buttonStage = Number(button.dataset.workflowStep) as WorkflowStage;
      const isActive = buttonStage === stage;
      if (isActive) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
      button.dataset.state = isActive
        ? 'active'
        : this.completed.has(buttonStage) || buttonStage < stage
          ? 'complete'
          : 'pending';
    }
    this.options.onStageChange?.(stage);
    if (focus) {
      const heading = this.stages.find((panel) => !panel.hidden)?.querySelector<HTMLElement>('h2');
      if (heading) {
        heading.tabIndex = -1;
        requestAnimationFrame(() => {
          heading.focus({ preventScroll: true });
          window.setTimeout(() => window.scrollTo({ top: 0, behavior: 'auto' }), 50);
        });
      }
    }
  }

  markComplete(stage: WorkflowStage): void {
    this.completed.add(stage);
    const button = this.stepButtons.find((candidate) => Number(candidate.dataset.workflowStep) === stage);
    if (button && stage !== this.currentStage) button.dataset.state = 'complete';
  }

  setStepSummary(stage: WorkflowStage, summary: string): void {
    const target = document.getElementById(`workflowStep${stage}Summary`);
    if (target) target.textContent = summary;
  }

  private bindPhotoSourceTabs(): void {
    const tabs = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-photo-source-tab]'));
    const panels = Array.from(document.querySelectorAll<HTMLElement>('[data-photo-source-panel]'));
    const selectTab = (tab: HTMLButtonElement, focus = false) => {
      const source = tab.dataset.photoSourceTab;
      for (const candidate of tabs) {
        const selected = candidate === tab;
        candidate.setAttribute('aria-selected', String(selected));
        candidate.tabIndex = selected ? 0 : -1;
      }
      for (const panel of panels) panel.hidden = panel.dataset.photoSourcePanel !== source;
      if (focus) tab.focus();
    };
    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => selectTab(tab));
      tab.addEventListener('keydown', (event) => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        const nextIndex =
          event.key === 'Home'
            ? 0
            : event.key === 'End'
              ? tabs.length - 1
              : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
        selectTab(tabs[nextIndex], true);
      });
    });
    if (tabs[0]) selectTab(tabs[0]);
  }
}
