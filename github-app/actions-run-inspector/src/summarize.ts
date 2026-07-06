interface WorkflowRun {
  id: number;
  status: string | null;
  conclusion: string | null;
  html_url: string;
  created_at: string;
  updated_at: string;
}

interface WorkflowJob {
  name: string;
  status: string;
  conclusion: string | null;
}

export function summarize(run: WorkflowRun, jobs: WorkflowJob[]) {
  const failedJobs = jobs.filter(j => j.conclusion === "failure");

  const duration =
    (new Date(run.updated_at).getTime() -
      new Date(run.created_at).getTime()) / 1000;

  return {
    run: {
      id: run.id,
      status: run.status,
      conclusion: run.conclusion,
      url: run.html_url
    },
    summary: {
      total_jobs: jobs.length,
      failed_jobs: failedJobs.length,
      duration_seconds: Math.round(duration)
    },
    jobs: jobs.map(j => ({
      name: j.name,
      status: j.status,
      conclusion: j.conclusion
    })),
    classification: classify(run, failedJobs)
  };
}

function classify(run: WorkflowRun, failedJobs: WorkflowJob[]) {
  if (run.conclusion === "success") return "success";

  if (failedJobs.some(j => j.name.toLowerCase().includes("test"))) {
    return "test_failure";
  }

  if (failedJobs.some(j => j.name.toLowerCase().includes("build"))) {
    return "build_failure";
  }

  if (run.conclusion === "cancelled") return "cancelled";

  return "unknown";
}
