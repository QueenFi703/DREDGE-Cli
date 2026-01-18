"""
GitHub Event Handler for DREDGE MCP Integration
Processes GitHub events and generates MCP responses
"""
import json
import sys
from typing import Any, Dict, Optional


class GitHubEventHandler:
    """Handles GitHub events and generates appropriate responses"""
    
    def __init__(self, event_name: str, event_payload: Dict[str, Any], 
                 ref: str, repo: str, sha: str):
        self.event_name = event_name
        self.event_payload = event_payload
        self.ref = ref
        self.repo = repo
        self.sha = sha
    
    def process(self) -> Dict[str, Any]:
        """Process the GitHub event and generate a response"""
        handler_method = getattr(self, f"_handle_{self.event_name}", self._handle_default)
        return handler_method()
    
    def _handle_push(self) -> Dict[str, Any]:
        """Handle push events"""
        commits = self.event_payload.get("commits", [])
        commit_count = len(commits)
        branch = self.ref.replace("refs/heads/", "")
        
        # Check if this is a Dependabot push
        is_dependabot = any(
            commit.get("author", {}).get("username") == "dependabot[bot]"
            for commit in commits
        )
        
        message = f"🔮 **DREDGE MCP**: Detected {commit_count} commit(s) on `{branch}`"
        
        if is_dependabot:
            message += "\n\n🤖 **DEPENDADREDGEABOT** update detected! Analyzing dependency changes..."
            analysis = self._analyze_dependabot_commits(commits)
            message += f"\n\n{analysis}"
        
        return {
            "status": "success",
            "event": "push",
            "branch": branch,
            "commits": commit_count,
            "comment": message,
            "is_dependabot": is_dependabot
        }
    
    def _handle_pull_request(self) -> Dict[str, Any]:
        """Handle pull request events"""
        pr = self.event_payload.get("pull_request", {})
        action = self.event_payload.get("action", "")
        pr_number = pr.get("number", 0)
        pr_title = pr.get("title", "")
        pr_user = pr.get("user", {}).get("login", "unknown")
        
        # Check if this is a Dependabot PR
        is_dependabot = pr_user == "dependabot[bot]"
        
        message = f"🔮 **DREDGE MCP**: PR #{pr_number} `{action}`\n\n**Title**: {pr_title}\n**Author**: {pr_user}"
        
        if is_dependabot:
            message += "\n\n🤖 **DEPENDADREDGEABOT** PR detected! Analyzing dependencies..."
            # Extract dependency information from PR title
            dep_analysis = self._analyze_dependabot_pr(pr)
            message += f"\n\n{dep_analysis}"
        else:
            message += "\n\n✨ DREDGE is monitoring this PR for insights."
        
        return {
            "status": "success",
            "event": "pull_request",
            "action": action,
            "pr_number": pr_number,
            "comment": message,
            "is_dependabot": is_dependabot
        }
    
    def _handle_issue_comment(self) -> Dict[str, Any]:
        """Handle issue comment events"""
        issue = self.event_payload.get("issue", {})
        comment = self.event_payload.get("comment", {})
        action = self.event_payload.get("action", "")
        
        issue_number = issue.get("number", 0)
        comment_body = comment.get("body", "")
        comment_user = comment.get("user", {}).get("login", "unknown")
        
        # Check if DREDGE is mentioned
        is_mentioned = "dredge" in comment_body.lower() or "@dredge" in comment_body.lower()
        
        message = None
        if is_mentioned:
            message = f"🔮 **DREDGE MCP**: Acknowledged mention in issue #{issue_number}\n\n"
            message += f"Processing request from @{comment_user}..."
        
        return {
            "status": "success",
            "event": "issue_comment",
            "action": action,
            "issue_number": issue_number,
            "comment": message,
            "is_mentioned": is_mentioned
        }
    
    def _handle_workflow_dispatch(self) -> Dict[str, Any]:
        """Handle manual workflow dispatch events"""
        inputs = self.event_payload.get("inputs", {})
        
        message = "🔮 **DREDGE MCP**: Manual workflow triggered\n\n"
        message += "DREDGE MCP is ready to respond to your prompts."
        
        return {
            "status": "success",
            "event": "workflow_dispatch",
            "inputs": inputs,
            "comment": message
        }
    
    def _handle_default(self) -> Dict[str, Any]:
        """Handle any other event type"""
        return {
            "status": "success",
            "event": self.event_name,
            "comment": f"🔮 **DREDGE MCP**: Received `{self.event_name}` event"
        }
    
    def _analyze_dependabot_commits(self, commits: list) -> str:
        """Analyze Dependabot commits"""
        analysis = "### Dependency Analysis\n\n"
        
        for commit in commits:
            message = commit.get("message", "")
            if "bump" in message.lower() or "update" in message.lower():
                analysis += f"- {message}\n"
        
        analysis += "\n✅ Dependency updates detected. DEPENDADREDGEABOT is keeping your code secure and up-to-date."
        return analysis
    
    def _analyze_dependabot_pr(self, pr: Dict[str, Any]) -> str:
        """Analyze Dependabot PR"""
        title = pr.get("title", "")
        body = pr.get("body", "")
        
        analysis = "### Dependency Update Analysis\n\n"
        analysis += f"**Update**: {title}\n\n"
        
        # Check for security updates
        if "security" in body.lower() or "vulnerability" in body.lower():
            analysis += "🔐 **Security Update**: This PR includes security fixes. Recommend immediate review and merge.\n\n"
        
        # Check for major version updates
        if "major" in body.lower():
            analysis += "⚠️ **Major Version**: This is a major version update. Review breaking changes carefully.\n\n"
        
        # Check for Python/Swift/Docker updates
        if any(eco in title.lower() for eco in ["python", "pip"]):
            analysis += "🐍 **Python Ecosystem**: DREDGE core dependencies affected.\n"
        elif any(eco in title.lower() for eco in ["swift"]):
            analysis += "🍎 **Swift Ecosystem**: DREDGE CLI native layer affected.\n"
        elif any(eco in title.lower() for eco in ["docker", "github-actions"]):
            analysis += "⚙️ **Infrastructure**: CI/CD or containerization affected.\n"
        
        analysis += "\n✨ DEPENDADREDGEABOT philosophy: *Be Literal. Be Philosophical. Be DEPENDADREDGEABOT.*"
        return analysis


def process_github_event(event_name: str, event_payload: str, ref: str, 
                         repo: str, sha: str) -> Dict[str, Any]:
    """
    Process a GitHub event and return a response
    
    Args:
        event_name: Name of the GitHub event (push, pull_request, etc.)
        event_payload: JSON string of the event payload
        ref: Git reference (branch/tag)
        repo: Repository name
        sha: Commit SHA
    
    Returns:
        Dictionary with response data
    """
    try:
        payload = json.loads(event_payload)
    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "error": f"Invalid JSON payload: {str(e)}"
        }
    
    handler = GitHubEventHandler(event_name, payload, ref, repo, sha)
    return handler.process()


def main():
    """CLI entry point for github-event command"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process GitHub events with DREDGE MCP"
    )
    parser.add_argument("--event", required=True, help="GitHub event name")
    parser.add_argument("--payload", required=True, help="GitHub event payload (JSON)")
    parser.add_argument("--ref", required=True, help="Git reference")
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument("--sha", required=True, help="Commit SHA")
    parser.add_argument("--out", default="out.json", help="Output file path")
    
    args = parser.parse_args()
    
    result = process_github_event(
        args.event,
        args.payload,
        args.ref,
        args.repo,
        args.sha
    )
    
    # Write output to file
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    
    # Print to stdout
    print(json.dumps(result, indent=2))
    
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
