# DREDGE Interactive Studio - Quick Start Guide

## 🚀 Starting the Web Server

The DREDGE Interactive Studio runs on **http://127.0.0.1:8000** with all credits integrated.

### Method 1: Using DREDGE CLI (Recommended)

```powershell
# Start on default port 8000
dredge interactive

# Or explicitly specify host and port
dredge interactive --host 127.0.0.1 --port 8000

# With auto-reload enabled (for development)
dredge interactive --port 8000 --reload
```

### Method 2: Direct Python Command

```powershell
# Start the interactive API server
python -m dredge interactive --port 8000

# Or with debug mode
python -m dredge interactive --port 8000 --debug
```

### Method 3: From Python Script

```python
from src.dredge.interactive_api import app
import uvicorn

# Run the server
uvicorn.run(
	app,
	host="127.0.0.1",
	port=8000,
	reload=True  # Auto-reload on file changes
)
```

## 🌐 Accessing the Web UI

Once started, open your browser and go to:

```
http://127.0.0.1:8000
```

## 📍 Credits Footer Location

The credits footer appears at the **bottom of the page** and displays:

```
DREDGE • Created by QueenFi703 • Maintained by Dredge Agent 
	   • Security: QueenFi703 & Dredge Agent • 🔗 GitHub
```

### Footer Features

- **Responsive Design**: Adapts to different screen sizes
- **GitHub Link**: Clickable link to the repository
- **Styled Text**: Matches DREDGE's dark theme with cyan accents
- **Professional Look**: Integrated with the overall UI design

## 🔍 What You'll See

### Page Structure
```
┌─────────────────────────────────────┐
│     DREDGE Interactive Studio        │
│                                      │
│        (Main UI Components)          │
│        - REPL Terminal               │
│        - Configuration Wizard        │
│        - Test Panel                  │
│        - Debug Tools                 │
│                                      │
│ ✓ Ready              12:34:56 PM     │  ← Status Bar
├─────────────────────────────────────┤
│ Credits Footer (with QueenFi703,    │  ← NEW: Credits Footer
│  Dredge Agent, & GitHub link)       │
└─────────────────────────────────────┘
```

## ⚙️ Server Details

| Property | Value |
|----------|-------|
| **Framework** | FastAPI + Uvicorn |
| **Default Host** | 127.0.0.1 (localhost) |
| **Default Port** | 8000 |
| **Web UI Location** | src/dredge/web_ui_html.py |
| **API Module** | src/dredge/interactive_api.py |

## 🔧 Available CLI Options

```powershell
dredge interactive --help
```

Options:
- `--host` - Host to bind to (default: 0.0.0.0)
- `--port` - Port to listen on (default: 8000)
- `--reload` - Enable auto-reload on file changes
- `--config` - Path to custom config file

## 🐛 Troubleshooting

### Port Already in Use

If port 8000 is already in use:
```powershell
# Use a different port
dredge interactive --port 8001

# Then access at: http://127.0.0.1:8001
```

### Import Errors

If you get import errors, ensure dependencies are installed:
```powershell
pip install -r requirements.txt
```

### CSS/JavaScript Not Loading

Clear browser cache (Ctrl+Shift+Delete) and reload the page.

## 📋 npm Installation (for GitHub App)

Before deploying the GitHub App Inspector, install dependencies:

```powershell
cd github-app
npm install

# This will install the updated versions:
# - @octokit/rest@^21.1.0
# - @octokit/auth-app@^7.2.0
```

## 🎯 What Changed

The integration adds:
1. ✅ Credits footer with proper attribution
2. ✅ Updated Octokit dependencies (fixes deprecation warnings)
3. ✅ Security improvements
4. ✅ Professional footer styling

## 📞 Support

For issues or questions:
- GitHub: https://github.com/QueenFi703/DREDGE-Cli
- Documentation: See docs/ directory

---

**Ready to go!** Start the server and visit http://127.0.0.1:8000 🚀
