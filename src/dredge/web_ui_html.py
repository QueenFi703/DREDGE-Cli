"""
Web UI Server - Serves React application for Interactive DREDGE
"""

import json
from pathlib import Path


def get_index_html() -> str:
    """Get the main index.html for the web UI"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interactive DREDGE Studio</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/atom-one-dark.min.css">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            :root {
                --primary: #0066cc;
                --secondary: #00d9ff;
                --dark: #1a1a1a;
                --darker: #0f0f0f;
                --border: #333;
                --text: #e0e0e0;
                --success: #00cc00;
                --error: #ff3333;
                --warning: #ffaa00;
            }

            body {
                font-family: 'Segoe UI', 'SF Mono', 'Monaco', 'Inconsolata', monospace;
                background: var(--dark);
                color: var(--text);
                overflow: hidden;
            }

            html, body, #app {
                width: 100%;
                height: 100%;
            }

            .container {
                display: flex;
                height: 100vh;
                width: 100%;
            }

            .sidebar {
                width: 250px;
                background: var(--darker);
                border-right: 1px solid var(--border);
                display: flex;
                flex-direction: column;
                overflow-y: auto;
            }

            .sidebar-header {
                padding: 20px;
                border-bottom: 1px solid var(--border);
                font-weight: bold;
                color: var(--secondary);
                font-size: 18px;
            }

            .sidebar-section {
                padding: 15px;
                border-bottom: 1px solid var(--border);
            }

            .sidebar-section-title {
                font-size: 12px;
                font-weight: bold;
                color: var(--secondary);
                margin-bottom: 10px;
                text-transform: uppercase;
            }

            .sidebar-item {
                padding: 10px;
                cursor: pointer;
                border-radius: 4px;
                margin-bottom: 5px;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .sidebar-item:hover {
                background: var(--border);
                padding-left: 15px;
            }

            .sidebar-item.active {
                background: var(--primary);
                color: white;
            }

            .main-content {
                flex: 1;
                display: flex;
                flex-direction: column;
            }

            .header {
                background: var(--darker);
                border-bottom: 1px solid var(--border);
                padding: 15px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .header-title {
                font-size: 20px;
                font-weight: bold;
                color: var(--secondary);
            }

            .header-actions {
                display: flex;
                gap: 10px;
            }

            .content {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
            }

            .panel {
                display: none;
            }

            .panel.active {
                display: block;
            }

            /* REPL Panel */
            .repl-container {
                display: flex;
                flex-direction: column;
                gap: 10px;
                height: 100%;
            }

            .repl-output {
                flex: 1;
                background: var(--darker);
                border: 1px solid var(--border);
                border-radius: 4px;
                padding: 15px;
                overflow-y: auto;
                font-size: 13px;
                font-family: 'SF Mono', monospace;
            }

            .repl-output-line {
                margin-bottom: 5px;
                word-wrap: break-word;
            }

            .repl-command {
                color: var(--secondary);
                margin-bottom: 5px;
            }

            .repl-result {
                color: var(--success);
                margin-bottom: 5px;
                margin-left: 20px;
            }

            .repl-error {
                color: var(--error);
                margin-bottom: 5px;
                margin-left: 20px;
            }

            .repl-input-group {
                display: flex;
                gap: 10px;
            }

            .repl-input {
                flex: 1;
                padding: 10px;
                background: var(--darker);
                border: 1px solid var(--border);
                border-radius: 4px;
                color: var(--text);
                font-family: 'SF Mono', monospace;
                font-size: 13px;
            }

            .repl-input:focus {
                outline: none;
                border-color: var(--primary);
            }

            .btn {
                padding: 10px 15px;
                background: var(--primary);
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                gap: 5px;
            }

            .btn:hover {
                background: var(--secondary);
                color: var(--dark);
            }

            .btn:active {
                transform: scale(0.95);
            }

            .btn-small {
                padding: 5px 10px;
                font-size: 12px;
            }

            /* Wizard Panel */
            .wizard-container {
                max-width: 600px;
            }

            .wizard-steps {
                display: flex;
                gap: 20px;
                margin-bottom: 30px;
                overflow-x: auto;
                padding-bottom: 10px;
            }

            .wizard-step {
                display: flex;
                flex-direction: column;
                align-items: center;
                min-width: 100px;
                cursor: pointer;
            }

            .wizard-step-number {
                width: 40px;
                height: 40px;
                background: var(--border);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                margin-bottom: 10px;
                transition: all 0.2s;
            }

            .wizard-step.active .wizard-step-number {
                background: var(--primary);
                color: white;
            }

            .wizard-step.completed .wizard-step-number {
                background: var(--success);
                color: white;
            }

            .wizard-step-title {
                font-size: 12px;
                text-align: center;
            }

            .wizard-form {
                background: var(--darker);
                border: 1px solid var(--border);
                border-radius: 4px;
                padding: 20px;
                margin-bottom: 20px;
            }

            .form-group {
                margin-bottom: 20px;
            }

            .form-label {
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
                font-size: 14px;
            }

            .form-control {
                width: 100%;
                padding: 10px;
                background: var(--dark);
                border: 1px solid var(--border);
                border-radius: 4px;
                color: var(--text);
                font-size: 13px;
                font-family: monospace;
            }

            .form-control:focus {
                outline: none;
                border-color: var(--primary);
            }

            textarea.form-control {
                min-height: 100px;
                resize: vertical;
            }

            .form-check {
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .form-check input[type="checkbox"] {
                width: 18px;
                height: 18px;
                cursor: pointer;
            }

            .wizard-buttons {
                display: flex;
                gap: 10px;
                justify-content: space-between;
            }

            /* Test Panel */
            .test-container {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }

            .test-list {
                background: var(--darker);
                border: 1px solid var(--border);
                border-radius: 4px;
                overflow: hidden;
            }

            .test-item {
                padding: 15px;
                border-bottom: 1px solid var(--border);
                display: flex;
                justify-content: space-between;
                align-items: center;
                cursor: pointer;
                transition: all 0.2s;
            }

            .test-item:hover {
                background: var(--border);
            }

            .test-item.passed {
                border-left: 3px solid var(--success);
            }

            .test-item.failed {
                border-left: 3px solid var(--error);
            }

            .test-item.skipped {
                border-left: 3px solid var(--warning);
            }

            .test-results {
                background: var(--darker);
                border: 1px solid var(--border);
                border-radius: 4px;
                padding: 15px;
                max-height: 300px;
                overflow-y: auto;
            }

            .test-result-item {
                padding: 10px;
                margin-bottom: 10px;
                background: var(--dark);
                border-left: 3px solid var(--border);
                border-radius: 2px;
            }

            .test-result-item.passed {
                border-left-color: var(--success);
            }

            .test-result-item.failed {
                border-left-color: var(--error);
            }

            .test-result-status {
                font-weight: bold;
                margin-bottom: 5px;
            }

            .test-result-output {
                font-size: 12px;
                color: #aaa;
                margin-top: 5px;
            }

            /* Swift Dependencies Panel */
            .deps-container {
                display: grid;
                grid-template-columns: minmax(260px, 360px) 1fr;
                gap: 20px;
                height: 100%;
            }

            .deps-actions {
                display: flex;
                flex-direction: column;
                gap: 12px;
            }

            .deps-action {
                background: var(--darker);
                border: 1px solid var(--border);
                border-radius: 4px;
                padding: 14px;
            }

            .deps-action-title {
                font-weight: bold;
                color: var(--secondary);
                margin-bottom: 6px;
            }

            .deps-action-path {
                font-size: 12px;
                color: #aaa;
                margin-bottom: 12px;
                font-family: 'SF Mono', monospace;
            }

            .deps-output {
                background: var(--darker);
                border: 1px solid var(--border);
                border-radius: 4px;
                padding: 15px;
                overflow-y: auto;
                white-space: pre-wrap;
                font-size: 12px;
                font-family: 'SF Mono', monospace;
                min-height: 420px;
            }

            .deps-status {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 12px;
                font-size: 13px;
            }

            .deps-status-dot {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: var(--border);
            }

            .deps-status.success .deps-status-dot {
                background: var(--success);
            }

            .deps-status.failed .deps-status-dot {
                background: var(--error);
            }

            /* Debug Panel */
            .debug-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                height: 100%;
            }

            .debug-panel {
                background: var(--darker);
                border: 1px solid var(--border);
                border-radius: 4px;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }

            .debug-panel-title {
                padding: 10px 15px;
                background: var(--dark);
                border-bottom: 1px solid var(--border);
                font-weight: bold;
                font-size: 13px;
            }

            .debug-panel-content {
                flex: 1;
                overflow-y: auto;
                padding: 15px;
                font-size: 12px;
                font-family: monospace;
            }

            .breakpoint-list {
                list-style: none;
            }

            .breakpoint-item {
                padding: 8px;
                background: var(--dark);
                border-radius: 2px;
                margin-bottom: 5px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .breakpoint-item.enabled {
                border-left: 3px solid var(--warning);
            }

            .breakpoint-item.disabled {
                border-left: 3px solid var(--border);
                opacity: 0.6;
            }

            /* Tabs */
            .tabs {
                display: flex;
                gap: 0;
                border-bottom: 1px solid var(--border);
                margin-bottom: 20px;
            }

            .tab {
                padding: 10px 20px;
                cursor: pointer;
                border-bottom: 2px solid transparent;
                transition: all 0.2s;
                font-size: 14px;
            }

            .tab:hover {
                color: var(--secondary);
            }

            .tab.active {
                border-bottom-color: var(--primary);
                color: var(--primary);
            }

            .tab-content {
                display: none;
            }

            .tab-content.active {
                display: block;
            }

            /* Status Bar */
            .status-bar {
                background: var(--darker);
                border-top: 1px solid var(--border);
                padding: 10px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 12px;
            }

            .status-item {
                display: flex;
                align-items: center;
                gap: 5px;
            }

            .status-indicator {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: var(--success);
            }

            .status-indicator.error {
                background: var(--error);
            }

            /* Credits Footer */
            .credits-footer {
                background: var(--darker);
                border-top: 1px solid var(--border);
                padding: 8px 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 11px;
                color: var(--text);
                min-height: 30px;
            }

            .credits-content {
                display: flex;
                align-items: center;
                gap: 12px;
                flex-wrap: wrap;
                justify-content: center;
            }

            .credits-label {
                font-weight: bold;
                color: var(--secondary);
                font-size: 12px;
                text-transform: uppercase;
            }

            .credits-author,
            .credits-agent,
            .credits-security {
                color: var(--text);
            }

            .credits-author strong,
            .credits-agent strong,
            .credits-security strong {
                color: var(--secondary);
                font-weight: 600;
            }

            .credits-separator {
                color: var(--border);
            }

            .credits-link {
                color: var(--secondary);
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 5px;
                transition: color 0.2s;
            }

            .credits-link:hover {
                color: var(--primary);
                text-decoration: underline;
            }

            /* Scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }

            ::-webkit-scrollbar-track {
                background: var(--darker);
            }

            ::-webkit-scrollbar-thumb {
                background: var(--border);
                border-radius: 4px;
            }

            ::-webkit-scrollbar-thumb:hover {
                background: var(--primary);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Sidebar -->
            <div class="sidebar">
                <div class="sidebar-header">
                    <i class="fas fa-code"></i> DREDGE Studio
                </div>

                <div class="sidebar-section">
                    <div class="sidebar-section-title">Development</div>
                    <div class="sidebar-item active" onclick="switchPanel('repl')">
                        <i class="fas fa-terminal"></i> REPL Console
                    </div>
                    <div class="sidebar-item" onclick="switchPanel('debug')">
                        <i class="fas fa-bug"></i> Debugger
                    </div>
                    <div class="sidebar-item" onclick="switchPanel('tests')">
                        <i class="fas fa-check-circle"></i> Tests
                    </div>
                </div>

                <div class="sidebar-section">
                    <div class="sidebar-section-title">Configuration</div>
                    <div class="sidebar-item" onclick="switchPanel('wizard')">
                        <i class="fas fa-wand-magic-sparkles"></i> Setup Wizard
                    </div>
                    <div class="sidebar-item" onclick="switchPanel('config')">
                        <i class="fas fa-cog"></i> Settings
                    </div>
                </div>

                <div class="sidebar-section">
                    <div class="sidebar-section-title">Build</div>
                    <div class="sidebar-item" onclick="switchPanel('swift-deps')">
                        <i class="fas fa-diagram-project"></i> Swift Dependencies
                    </div>
                    <div class="sidebar-item" onclick="buildSwift()">
                        <i class="fas fa-cube"></i> Build Swift
                    </div>
                    <div class="sidebar-item" onclick="buildPython()">
                        <i class="fas fa-snake"></i> Build Python
                    </div>
                </div>
            </div>

            <!-- Main Content -->
            <div class="main-content">
                <!-- Header -->
                <div class="header">
                    <div class="header-title" id="panel-title">REPL Console</div>
                    <div class="header-actions">
                        <button class="btn btn-small" onclick="clearOutput()">
                            <i class="fas fa-trash"></i> Clear
                        </button>
                        <button class="btn btn-small" onclick="exportSession()">
                            <i class="fas fa-download"></i> Export
                        </button>
                    </div>
                </div>

                <!-- Content -->
                <div class="content">
                    <!-- REPL Panel -->
                    <div id="repl" class="panel active">
                        <div class="repl-container">
                            <div class="repl-output" id="repl-output"></div>
                            <div class="repl-input-group">
                                <input type="text" class="repl-input" id="repl-input" placeholder="Enter Swift or Python command...">
                                <button class="btn" onclick="executeREPLCommand()">
                                    <i class="fas fa-play"></i> Run
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- Wizard Panel -->
                    <div id="wizard" class="panel">
                        <div class="wizard-container">
                            <div class="wizard-steps" id="wizard-steps"></div>
                            <div class="wizard-form" id="wizard-form"></div>
                            <div class="wizard-buttons">
                                <button class="btn" onclick="previousStep()" id="prev-btn" style="display:none;">
                                    <i class="fas fa-chevron-left"></i> Previous
                                </button>
                                <button class="btn" onclick="nextStep()" id="next-btn">
                                    <i class="fas fa-chevron-right"></i> Next
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- Tests Panel -->
                    <div id="tests" class="panel">
                        <div class="test-container">
                            <div>
                                <button class="btn" onclick="runAllTests()">
                                    <i class="fas fa-play"></i> Run All Tests
                                </button>
                            </div>
                            <div class="test-results" id="test-results"></div>
                        </div>
                    </div>

                    <!-- Swift Dependencies Panel -->
                    <div id="swift-deps" class="panel">
                        <div class="deps-container">
                            <div class="deps-actions">
                                <div class="deps-action">
                                    <div class="deps-action-title">Resolve Packages</div>
                                    <div class="deps-action-path">Package.swift + swift/Package.swift</div>
                                    <button class="btn" onclick="resolveSwiftDependencies()">
                                        <i class="fas fa-link"></i> Resolve
                                    </button>
                                </div>
                                <div class="deps-action">
                                    <div class="deps-action-title">Build Local DREDGE</div>
                                    <div class="deps-action-path">swift/DREDGE</div>
                                    <button class="btn" onclick="buildSwiftDependency()">
                                        <i class="fas fa-hammer"></i> Build Dependency
                                    </button>
                                </div>
                                <div class="deps-action">
                                    <div class="deps-action-title">Build Swift CLI</div>
                                    <div class="deps-action-path">swift</div>
                                    <button class="btn" onclick="buildSwift()">
                                        <i class="fas fa-cube"></i> Build Swift
                                    </button>
                                </div>
                                <div class="deps-action">
                                    <div class="deps-action-title">Package Graph</div>
                                    <div class="deps-action-path">swift package describe</div>
                                    <button class="btn" onclick="describeSwiftDependencies()">
                                        <i class="fas fa-list"></i> Describe
                                    </button>
                                </div>
                            </div>
                            <div>
                                <div class="deps-status" id="deps-status">
                                    <span class="deps-status-dot"></span>
                                    <span id="deps-status-text">Ready</span>
                                </div>
                                <div class="deps-output" id="deps-output">Swift dependency output will appear here.</div>
                            </div>
                        </div>
                    </div>

                    <!-- Debug Panel -->
                    <div id="debug" class="panel">
                        <div class="debug-container">
                            <div class="debug-panel">
                                <div class="debug-panel-title">Breakpoints</div>
                                <div class="debug-panel-content">
                                    <ul class="breakpoint-list" id="breakpoint-list"></ul>
                                </div>
                            </div>
                            <div class="debug-panel">
                                <div class="debug-panel-title">Variables</div>
                                <div class="debug-panel-content" id="variables-panel"></div>
                            </div>
                        </div>
                    </div>

                    <!-- Config Panel -->
                    <div id="config" class="panel">
                        <h3>Configuration</h3>
                        <div class="wizard-form" id="config-form"></div>
                    </div>
                </div>

                <!-- Status Bar -->
                <div class="status-bar">
                    <div class="status-item">
                        <div class="status-indicator"></div>
                        <span id="status-text">Ready</span>
                    </div>
                    <div class="status-item">
                        <span id="status-time"></span>
                    </div>
                </div>

                <!-- Credits Footer -->
                <div class="credits-footer">
                    <div class="credits-content">
                        <span class="credits-label">DREDGE</span>
                        <span class="credits-author">Created by <strong>QueenFi703</strong></span>
                        <span class="credits-separator">•</span>
                        <span class="credits-agent">Maintained by <strong>Dredge Agent</strong></span>
                        <span class="credits-separator">•</span>
                        <span class="credits-security">Security: <strong>QueenFi703 & Dredge Agent</strong></span>
                        <span class="credits-separator">•</span>
                        <a href="https://github.com/QueenFi703/DREDGE-Cli" class="credits-link" target="_blank">
                            <i class="fab fa-github"></i> GitHub
                        </a>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
        <script>
            let currentPanel = 'repl';
            let replSessionId = null;
            let currentWizardStep = 1;
            let ws = null;

            // Initialize on load
            async function init() {
                await createREPLSession();
                await loadWizardSteps();
                setupEventListeners();
                updateClock();
                setInterval(updateClock, 1000);
            }

            function setupEventListeners() {
                document.getElementById('repl-input').addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        executeREPLCommand();
                    }
                });
            }

            async function createREPLSession() {
                try {
                    const response = await fetch('/api/repl/sessions?language=swift', { method: 'POST' });
                    const data = await response.json();
                    replSessionId = data.session_id;
                    addOutputLine(`Session created: ${replSessionId}`, 'repl-result');
                } catch (e) {
                    console.error('Failed to create REPL session:', e);
                    addOutputLine('Failed to create REPL session', 'repl-error');
                }
            }

            async function executeREPLCommand() {
                const input = document.getElementById('repl-input');
                const command = input.value.trim();
                if (!command) return;

                addOutputLine(`> ${command}`, 'repl-command');
                input.value = '';
                setStatus('Executing...', 'error');

                try {
                    const response = await fetch('/api/repl/execute', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            command: command,
                            session_id: replSessionId,
                            language: 'swift'
                        })
                    });

                    const data = await response.json();
                    if (data.output) {
                        addOutputLine(data.output, 'repl-result');
                    }
                    if (data.error) {
                        addOutputLine(data.error, 'repl-error');
                    }
                    setStatus(`Ready (${data.execution_time.toFixed(3)}s)`);
                } catch (e) {
                    addOutputLine(`Error: ${e.message}`, 'repl-error');
                    setStatus('Ready');
                }
            }

            function addOutputLine(text, className = 'repl-output-line') {
                const output = document.getElementById('repl-output');
                const line = document.createElement('div');
                line.className = className;
                line.textContent = text;
                output.appendChild(line);
                output.scrollTop = output.scrollHeight;
            }

            function setDepsStatus(text, status = '') {
                const statusEl = document.getElementById('deps-status');
                const textEl = document.getElementById('deps-status-text');
                statusEl.className = `deps-status ${status}`;
                textEl.textContent = text;
            }

            function writeDepsOutput(text) {
                const output = document.getElementById('deps-output');
                output.textContent = text || '(no output)';
                output.scrollTop = output.scrollHeight;
            }

            function formatCommandResult(data) {
                const lines = [];
                if (data.status) lines.push(`status: ${data.status}`);
                if (data.command) lines.push(`command: ${data.command}`);
                if (data.cwd) lines.push(`cwd: ${data.cwd}`);
                if (data.return_code !== undefined) lines.push(`return code: ${data.return_code}`);
                if (data.steps) {
                    data.steps.forEach(step => {
                        lines.push('');
                        lines.push(`[${step.name}] ${step.status}`);
                        if (step.command) lines.push(`command: ${step.command}`);
                        if (step.cwd) lines.push(`cwd: ${step.cwd}`);
                        if (step.output) lines.push(step.output);
                        if (step.errors && step.errors.length) lines.push(step.errors.join('\\n'));
                    });
                }
                if (data.output) {
                    lines.push('');
                    lines.push(data.output);
                }
                if (data.errors && data.errors.length) {
                    lines.push('');
                    lines.push(data.errors.join('\\n'));
                }
                return lines.join('\\n');
            }

            async function runSwiftDependencyAction(label, url, method = 'POST') {
                switchPanel('swift-deps');
                setDepsStatus(`${label}...`, '');
                writeDepsOutput(`Running ${label}...`);
                setStatus(`${label}...`, 'error');

                try {
                    const response = await fetch(url, { method });
                    const data = await response.json();
                    const ok = response.ok && data.status !== 'failed';
                    setDepsStatus(ok ? `${label} complete` : `${label} failed`, ok ? 'success' : 'failed');
                    writeDepsOutput(formatCommandResult(data));
                } catch (e) {
                    setDepsStatus(`${label} failed`, 'failed');
                    writeDepsOutput(`Error: ${e.message}`);
                }
                setStatus('Ready');
            }

            function resolveSwiftDependencies() {
                runSwiftDependencyAction('Resolving Swift dependencies', '/api/swift/dependencies/resolve');
            }

            function buildSwiftDependency() {
                runSwiftDependencyAction('Building DREDGE Swift dependency', '/api/swift/dependencies/build');
            }

            function describeSwiftDependencies() {
                runSwiftDependencyAction('Describing Swift dependency graph', '/api/swift/dependencies/describe', 'GET');
            }

            async function loadWizardSteps() {
                try {
                    const response = await fetch('/api/wizard/steps');
                    const steps = await response.json();
                    renderWizardSteps(steps);
                } catch (e) {
                    console.error('Failed to load wizard steps:', e);
                }
            }

            function renderWizardSteps(steps) {
                const container = document.getElementById('wizard-steps');
                container.innerHTML = steps.map(step => `
                    <div class="wizard-step ${step.step_id === currentWizardStep ? 'active' : ''}" onclick="goToStep(${step.step_id})">
                        <div class="wizard-step-number">${step.step_id}</div>
                        <div class="wizard-step-title">${step.title}</div>
                    </div>
                `).join('');
                renderWizardForm(steps[currentWizardStep - 1]);
            }

            function renderWizardForm(step) {
                const form = document.getElementById('wizard-form');
                form.innerHTML = `
                    <h3>${step.title}</h3>
                    <p>${step.description}</p>
                    <div id="form-fields"></div>
                `;

                const fieldsContainer = document.getElementById('form-fields');
                fieldsContainer.innerHTML = step.fields.map(field => {
                    if (field.type === 'checkbox') {
                        return `
                            <div class="form-group">
                                <div class="form-check">
                                    <input type="checkbox" id="${field.name}" ${field.value ? 'checked' : ''}>
                                    <label class="form-label" for="${field.name}">${field.name}</label>
                                </div>
                            </div>
                        `;
                    } else if (field.type === 'select') {
                        return `
                            <div class="form-group">
                                <label class="form-label">${field.name}</label>
                                <select class="form-control" id="${field.name}">
                                    ${field.options.map(opt => `<option>${opt}</option>`).join('')}
                                </select>
                            </div>
                        `;
                    } else {
                        return `
                            <div class="form-group">
                                <label class="form-label">${field.name}</label>
                                <input type="text" class="form-control" id="${field.name}" placeholder="${field.name}" value="${field.value || ''}">
                            </div>
                        `;
                    }
                }).join('');
            }

            function switchPanel(panelName) {
                // Hide all panels
                document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));

                // Show selected panel
                const panel = document.getElementById(panelName);
                if (panel) {
                    panel.classList.add('active');
                    document.getElementById('panel-title').textContent = 
                        panelName.charAt(0).toUpperCase() + panelName.slice(1);
                }

                currentPanel = panelName;
            }

            function clearOutput() {
                document.getElementById('repl-output').innerHTML = '';
            }

            async function buildSwift() {
                switchPanel('swift-deps');
                setStatus('Building Swift...', 'error');
                setDepsStatus('Building Swift package...', '');
                writeDepsOutput('Running swift build...');
                try {
                    const response = await fetch('/api/build/swift', { method: 'POST' });
                    const data = await response.json();
                    setDepsStatus(data.status === 'success' ? 'Swift build complete' : 'Swift build failed', data.status);
                    writeDepsOutput(formatCommandResult(data));
                    addOutputLine(`Build status: ${data.status}`, 'repl-result');
                    if (data.errors.length > 0) {
                        data.errors.forEach(err => addOutputLine(err, 'repl-error'));
                    }
                } catch (e) {
                    addOutputLine(`Build error: ${e.message}`, 'repl-error');
                }
                setStatus('Ready');
            }

            async function buildPython() {
                setStatus('Building Python...', 'error');
                try {
                    const response = await fetch('/api/build/python', { method: 'POST' });
                    const data = await response.json();
                    addOutputLine(`Build status: ${data.status}`, 'repl-result');
                } catch (e) {
                    addOutputLine(`Build error: ${e.message}`, 'repl-error');
                }
                setStatus('Ready');
            }

            function nextStep() {
                currentWizardStep++;
                // TODO: load next step
            }

            function previousStep() {
                if (currentWizardStep > 1) currentWizardStep--;
                // TODO: load previous step
            }

            function goToStep(step) {
                currentWizardStep = step;
                // TODO: load step
            }

            async function runAllTests() {
                setStatus('Running tests...', 'error');
                try {
                    const response = await fetch('/api/tests/run-all', { method: 'POST' });
                    const data = await response.json();
                    const resultsDiv = document.getElementById('test-results');
                    resultsDiv.innerHTML = data.results.map(r => `
                        <div class="test-result-item ${r.status}">
                            <div class="test-result-status">${r.test_name}: ${r.status.toUpperCase()}</div>
                            <div class="test-result-output">${r.output}</div>
                        </div>
                    `).join('');
                } catch (e) {
                    console.error('Test execution error:', e);
                }
                setStatus('Ready');
            }

            function exportSession() {
                addOutputLine('Exporting session...', 'repl-result');
                // TODO: implement export
            }

            function setStatus(text, className = '') {
                const status = document.getElementById('status-text');
                status.textContent = text;
                status.className = className;
            }

            function updateClock() {
                const now = new Date().toLocaleTimeString();
                document.getElementById('status-time').textContent = now;
            }

            // Initialize on load
            document.addEventListener('DOMContentLoaded', init);
        </script>
    </body>
    </html>
    """
