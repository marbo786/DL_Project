import { ArrowDownRight, ScanLine } from 'lucide-react';
import type { ReactNode } from 'react';
import { ActiveNavigation } from '@/components/active-navigation';
import { ThemeToggle } from '@/components/theme-toggle';
import { StaticResults } from '@/components/static-results';
import { ChartExplorer } from '@/components/chart-explorer';
import { DemoLoader } from '@/components/demo-loader';
import { ModelLimits } from '@/components/model-limits';
import Pipeline from '@/components/pipeline';
import { steps } from '@/lib/pipeline-steps';
import type { ResultsData } from '@/lib/results';

const repository = 'https://github.com/marbo786/Forgery_Lens';

export function Workbench({ evidence, data }: { evidence: ReactNode; data: ResultsData }) {
  return (
    <div className="preview workbench">
      <a className="skip-link" href="#main">Skip to main content</a>
      <div className="workbench-chrome" data-scroll-header>
        <div className="research-banner persistent-banner" role="note">
          Research prototype, not for real reimbursement decisions. Saved outputs from a frozen model that performs near chance.
        </div>
        <header className="masthead wrap workbench-header">
          <a className="wordmark" href="#main"><span className="wordmark-icon" aria-hidden="true"><ScanLine size={20} strokeWidth={1.7} /></span>ForgeryLens</a>
          <ActiveNavigation />
          <ThemeToggle />
        </header>
      </div>
      <main className="wrap workbench-main" id="main" tabIndex={-1}>
        <section className="hero" aria-labelledby="hero-title">
          <div className="hero-text">
            <h1 id="hero-title">Receipt edits,<br /><em>under review.</em></h1>
            <p>See the model's highest-scoring region and review decision on a fictional receipt.</p>
            <a className="control control-primary" href="#demo">Open the receipt studio <ArrowDownRight size={18} aria-hidden="true" /></a>
          </div>
          {evidence}
        </section>
        <section className="work-section" id="demo" aria-labelledby="demo-heading">
          <div className="section-heading"><h2 id="demo-heading" data-scroll-heading tabIndex={-1}>Receipt studio</h2><p>Choose a fictional receipt to inspect the recorded tile scores.</p></div>
          <DemoLoader />
        </section>
        <section className="work-section" id="results" aria-labelledby="results-heading">
          <div className="section-heading"><h2 id="results-heading" data-scroll-heading tabIndex={-1}>Research results</h2><p>See how the detector ranked edits across receipt sources, and how image stress affected performance.</p></div>
          <StaticResults data={data} />
          <ChartExplorer />
        </section>
        <section className="work-section" id="method" aria-labelledby="method-heading">
          <div className="section-heading"><h2 id="method-heading" data-scroll-heading tabIndex={-1}>How it works</h2></div>
          <details className="static-detail method-transcript"><summary>Read the five steps</summary>
            <ol>{steps.map(step => <li key={step.title}><strong>{step.title}</strong><p>{step.text}</p></li>)}</ol>
          </details>
          <Pipeline />
        </section>
        <section className="work-section" id="limits" aria-labelledby="limits-heading">
          <div className="section-heading"><h2 id="limits-heading" data-scroll-heading tabIndex={-1}>Limits and model card</h2></div>
          <ModelLimits data={data} />
        </section>
      </main>
      <footer className="wrap workbench-footer">
        <span>ForgeryLens · AI341 Deep Neural Networks</span><a href={repository}>Project repository</a>
      </footer>
    </div>
  );
}
