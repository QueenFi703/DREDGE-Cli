"""
Tests for GitHub Event Handler
"""
import json
import pytest
from dredge.github_event_handler import GitHubEventHandler, process_github_event


class TestGitHubEventHandler:
    """Test the GitHubEventHandler class"""
    
    def test_handle_push_event(self):
        """Test handling of push events"""
        event_payload = {
            "commits": [
                {
                    "message": "Update dependencies",
                    "author": {"username": "testuser"}
                }
            ]
        }
        
        handler = GitHubEventHandler(
            event_name="push",
            event_payload=event_payload,
            ref="refs/heads/main",
            repo="test/repo",
            sha="abc123"
        )
        
        result = handler.process()
        
        assert result["status"] == "success"
        assert result["event"] == "push"
        assert result["branch"] == "main"
        assert result["commits"] == 1
        assert "DREDGE MCP" in result["comment"]
        assert result["is_dependabot"] is False
    
    def test_handle_dependabot_push(self):
        """Test handling of Dependabot push events"""
        event_payload = {
            "commits": [
                {
                    "message": "Bump torch from 2.0.0 to 2.1.0",
                    "author": {"username": "dependabot[bot]"}
                }
            ]
        }
        
        handler = GitHubEventHandler(
            event_name="push",
            event_payload=event_payload,
            ref="refs/heads/dependabot/pip/torch-2.1.0",
            repo="test/repo",
            sha="abc123"
        )
        
        result = handler.process()
        
        assert result["status"] == "success"
        assert result["is_dependabot"] == True
        assert "DEPENDADREDGEABOT" in result["comment"]
        assert "Analyzing dependency changes" in result["comment"]
    
    def test_handle_pull_request_event(self):
        """Test handling of pull request events"""
        event_payload = {
            "action": "opened",
            "pull_request": {
                "number": 42,
                "title": "Add new feature",
                "user": {"login": "testuser"}
            }
        }
        
        handler = GitHubEventHandler(
            event_name="pull_request",
            event_payload=event_payload,
            ref="refs/pull/42/merge",
            repo="test/repo",
            sha="abc123"
        )
        
        result = handler.process()
        
        assert result["status"] == "success"
        assert result["event"] == "pull_request"
        assert result["action"] == "opened"
        assert result["pr_number"] == 42
        assert "DREDGE MCP" in result["comment"]
        assert result["is_dependabot"] is False
    
    def test_handle_dependabot_pr(self):
        """Test handling of Dependabot pull request events"""
        event_payload = {
            "action": "opened",
            "pull_request": {
                "number": 100,
                "title": "Bump numpy from 1.24.0 to 1.25.0",
                "user": {"login": "dependabot[bot]"},
                "body": "Bumps numpy from 1.24.0 to 1.25.0"
            }
        }
        
        handler = GitHubEventHandler(
            event_name="pull_request",
            event_payload=event_payload,
            ref="refs/pull/100/merge",
            repo="test/repo",
            sha="abc123"
        )
        
        result = handler.process()
        
        assert result["status"] == "success"
        assert result["is_dependabot"] == True
        assert "DEPENDADREDGEABOT" in result["comment"]
        assert "Analyzing dependencies" in result["comment"]
    
    def test_handle_dependabot_security_pr(self):
        """Test handling of Dependabot security update PRs"""
        event_payload = {
            "action": "opened",
            "pull_request": {
                "number": 101,
                "title": "Bump flask from 2.0.0 to 3.0.0",
                "user": {"login": "dependabot[bot]"},
                "body": "This update includes security fixes for vulnerabilities"
            }
        }
        
        handler = GitHubEventHandler(
            event_name="pull_request",
            event_payload=event_payload,
            ref="refs/pull/101/merge",
            repo="test/repo",
            sha="abc123"
        )
        
        result = handler.process()
        
        assert result["status"] == "success"
        assert result["is_dependabot"] == True
        assert "Security Update" in result["comment"]
    
    def test_handle_issue_comment_event(self):
        """Test handling of issue comment events"""
        event_payload = {
            "action": "created",
            "issue": {
                "number": 10
            },
            "comment": {
                "body": "Can @dredge help with this?",
                "user": {"login": "testuser"}
            }
        }
        
        handler = GitHubEventHandler(
            event_name="issue_comment",
            event_payload=event_payload,
            ref="refs/heads/main",
            repo="test/repo",
            sha="abc123"
        )
        
        result = handler.process()
        
        assert result["status"] == "success"
        assert result["event"] == "issue_comment"
        assert result["is_mentioned"] == True
        assert result["comment"] is not None
        assert "DREDGE MCP" in result["comment"]
    
    def test_handle_issue_comment_no_mention(self):
        """Test handling of issue comment without DREDGE mention"""
        event_payload = {
            "action": "created",
            "issue": {
                "number": 10
            },
            "comment": {
                "body": "This is a regular comment",
                "user": {"login": "testuser"}
            }
        }
        
        handler = GitHubEventHandler(
            event_name="issue_comment",
            event_payload=event_payload,
            ref="refs/heads/main",
            repo="test/repo",
            sha="abc123"
        )
        
        result = handler.process()
        
        assert result["status"] == "success"
        assert result["is_mentioned"] is False
        assert result["comment"] is None
    
    def test_handle_workflow_dispatch(self):
        """Test handling of workflow_dispatch events"""
        event_payload = {
            "inputs": {
                "custom_message": "Test message"
            }
        }
        
        handler = GitHubEventHandler(
            event_name="workflow_dispatch",
            event_payload=event_payload,
            ref="refs/heads/main",
            repo="test/repo",
            sha="abc123"
        )
        
        result = handler.process()
        
        assert result["status"] == "success"
        assert result["event"] == "workflow_dispatch"
        assert "DREDGE MCP" in result["comment"]
    
    def test_handle_unknown_event(self):
        """Test handling of unknown event types"""
        handler = GitHubEventHandler(
            event_name="unknown_event",
            event_payload={},
            ref="refs/heads/main",
            repo="test/repo",
            sha="abc123"
        )
        
        result = handler.process()
        
        assert result["status"] == "success"
        assert result["event"] == "unknown_event"


class TestProcessGitHubEvent:
    """Test the process_github_event function"""
    
    def test_valid_json_payload(self):
        """Test processing with valid JSON payload"""
        payload = json.dumps({"commits": []})
        
        result = process_github_event(
            event_name="push",
            event_payload=payload,
            ref="refs/heads/main",
            repo="test/repo",
            sha="abc123"
        )
        
        assert result["status"] == "success"
    
    def test_invalid_json_payload(self):
        """Test processing with invalid JSON payload"""
        result = process_github_event(
            event_name="push",
            event_payload="invalid json {",
            ref="refs/heads/main",
            repo="test/repo",
            sha="abc123"
        )
        
        assert result["status"] == "error"
        assert "Invalid JSON payload" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
