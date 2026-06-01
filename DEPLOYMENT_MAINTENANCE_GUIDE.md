# DREDGE Web UI Integration - Deployment & Maintenance Guide

## 📦 Deployment Steps

### Pre-Deployment Checklist

- [ ] All changes committed to git
- [ ] Dependencies updated (npm install done)
- [ ] Tests passing (if applicable)
- [ ] Documentation reviewed
- [ ] Browser testing completed
- [ ] Security review done

### Step 1: Update Dependencies

```powershell
# Navigate to github-app directory
cd github-app

# Install updated npm packages
npm install

# Verify installation
npm list @octokit/rest @octokit/auth-app
```

Expected output:
```
github-app@1.0.0
├── @octokit/auth-app@7.2.0
├── @octokit/rest@21.1.0
└── ...
```

### Step 2: Verify Changes

```powershell
# Run syntax checks
python -m py_compile src/dredge/web_ui_html.py

# Validate JSON
Get-Content github-app/package.json | ConvertFrom-Json
Get-Content github-app/actions-run-inspector/package.json | ConvertFrom-Json

# Check git status
git status
```

### Step 3: Local Testing

```powershell
# Start development server
dredge interactive --port 8000 --reload

# In another terminal, run tests (if available)
# cd tests && python -m pytest test_server.py
```

### Step 4: Browser Testing

Open `http://127.0.0.1:8000` and verify:

1. **Page Loads**
   - [ ] No console errors (F12)
   - [ ] No broken layouts
   - [ ] All UI elements visible

2. **Credits Footer**
   - [ ] Footer appears at bottom
   - [ ] "DREDGE" label visible in cyan
   - [ ] All names properly displayed
   - [ ] GitHub link visible

3. **Interactive Elements**
   - [ ] Hover over GitHub link
   - [ ] Color changes to blue
   - [ ] Underline appears
   - [ ] Click GitHub link
   - [ ] Opens QueenFi703/DREDGE-Cli in new tab

4. **Responsive Testing**
   - [ ] Resize browser window
   - [ ] Footer wraps correctly on mobile
   - [ ] Text remains readable
   - [ ] No overflow issues

### Step 5: Performance Testing

```powershell
# Check page load time
# Open browser DevTools (F12)
# Go to Network tab
# Reload page
# Check total load time (should be < 2s)
```

### Step 6: Commit & Deploy

```powershell
# Stage changes
git add -A

# Commit with proper message
git commit -m "feat: Add GitHub App Inspector credits to web UI

- Integrate credits footer with QueenFi703 & Dredge Agent attribution
- Update @octokit/rest to v21.1.0 (fixes DEP0169)
- Update @octokit/auth-app to v7.2.0 (security)
- Add responsive footer styling
- Update AUTHORS.md with security credits"

# Push to remote
git push origin feat/swift-docker-integration

# Create Pull Request (if using GitHub)
```

## 🔧 Maintenance Guide

### Regular Maintenance Tasks

#### Weekly
- Monitor error logs for any issues
- Check for Node.js deprecation warnings
- Verify footer displays correctly

#### Monthly
- Check for Octokit updates
  ```powershell
  npm outdated -g --depth=0
  ```
- Review security advisories
  ```powershell
  npm audit
  ```
- Update dependencies if security patches available
  ```powershell
  npm update
  ```

#### Quarterly
- Full dependency audit
- Performance review
- Browser compatibility testing
- Accessibility review

### Updating Octokit Dependencies

When new versions are released:

```powershell
# Check current versions
npm list @octokit/rest @octokit/auth-app

# Update to latest
npm install @octokit/rest@latest @octokit/auth-app@latest

# Test thoroughly before deploying
dredge interactive --port 8000 --reload

# Commit if working well
git add package.json package-lock.json
git commit -m "chore: update Octokit dependencies"
```

## 🐛 Troubleshooting & Issues

### Issue: Footer Not Displaying

**Symptoms**: Credits footer missing from web UI

**Solutions**:
1. Clear browser cache
   ```powershell
   # Clear browser cache (varies by browser)
   # Chrome: Ctrl+Shift+Delete
   # Firefox: Ctrl+Shift+Delete
   # Edge: Ctrl+Shift+Delete
   ```

2. Check browser console
   ```
   Press F12 → Console tab
   Look for JavaScript errors
   ```

3. Verify web_ui_html.py modification
   ```powershell
   # Search for credits-footer in the file
   Select-String -Path "src/dredge/web_ui_html.py" -Pattern "credits-footer"
   ```

4. Restart server
   ```powershell
   # Stop current server (Ctrl+C)
   # Restart
   dredge interactive --port 8000
   ```

### Issue: Styling Not Applied

**Symptoms**: Footer appears but with wrong colors/layout

**Solutions**:
1. Check CSS is loaded
   ```
   Right-click footer → Inspect
   Check Styles panel in DevTools
   ```

2. Clear CSS cache
   ```powershell
   # Hard refresh page
   Ctrl+Shift+R (most browsers)
   ```

3. Verify CSS variables
   ```
   DevTools → Console
   getComputedStyle(document.documentElement).getPropertyValue('--secondary')
   Should return: " #00d9ff"
   ```

### Issue: GitHub Link Not Working

**Symptoms**: Clicking GitHub link doesn't work or opens wrong URL

**Solutions**:
1. Check link in HTML
   ```powershell
   Select-String -Path "src/dredge/web_ui_html.py" -Pattern "github.com/QueenFi703"
   ```

2. Verify href attribute
   ```
   Should be: https://github.com/QueenFi703/DREDGE-Cli
   Target should be: _blank
   ```

3. Test in browser DevTools
   ```
   Right-click link → Inspect
   Hover and check URL in status bar
   ```

### Issue: Node.js Deprecation Warnings Still Present

**Symptoms**: Still seeing "DEP0169" warning

**Solutions**:
1. Verify npm packages updated
   ```powershell
   npm list @octokit/rest
   Should show: @octokit/rest@21.1.0
   ```

2. Clear node_modules and reinstall
   ```powershell
   Remove-Item -Recurse node_modules
   npm install
   ```

3. Check for other url.parse usage
   ```powershell
   Select-String -Recurse -Pattern "url\.parse" src/ --Include="*.js","*.ts"
   ```

## 📊 Monitoring

### Health Checks

Create a monitoring script to periodically check:

```powershell
# Check if server is running
$response = Invoke-WebRequest -Uri "http://127.0.0.1:8000" -ErrorAction SilentlyContinue

if ($response.StatusCode -eq 200) {
	Write-Host "✅ Server is healthy" -ForegroundColor Green
} else {
	Write-Host "❌ Server returned: $($response.StatusCode)" -ForegroundColor Red
}

# Check for deprecation warnings
$logs = dredge interactive --port 8000 2>&1
if ($logs -match "DEP0169") {
	Write-Host "⚠️  Deprecation warning found" -ForegroundColor Yellow
} else {
	Write-Host "✅ No deprecation warnings" -ForegroundColor Green
}
```

### Logging

Monitor these log locations:

```
Application Logs: ~/.dredge/logs/
Error Logs: ~/.dredge/logs/error.log
Node Logs: github-app/npm-debug.log
```

## 📈 Performance Optimization

### Current Performance

- Footer HTML: ~400 bytes
- Footer CSS: ~1.2 KB
- Additional Load Time: < 1ms
- Rendering Impact: Minimal

### Future Optimization Opportunities

1. **Minify CSS**
   - Reduce footer CSS to ~600 bytes

2. **Lazy Load Footer**
   - Load after main content (not critical)

3. **Cache Footer**
   - Cache rendered footer in session

4. **CDN Footer Resources**
   - Move Font Awesome icons to CDN

## 🔒 Security Maintenance

### Regular Security Checks

```powershell
# Weekly npm audit
npm audit

# Fix vulnerabilities
npm audit fix

# Check for outdated packages
npm outdated

# Verify no hardcoded secrets
Select-String -Recurse -Pattern "password|secret|token|key" src/ --Exclude="*.pyc"
```

### Dependency Review

Keep these up to date:

- [ ] @octokit/rest - Critical security updates
- [ ] @octokit/auth-app - Critical security updates
- [ ] Flask - Security patches
- [ ] FastAPI - Security patches
- [ ] uvicorn - Security patches

## 📞 Support & Escalation

### Common Issues & Escalation Path

```
User Reports Issue
	↓
Check Browser Compatibility
	↓
Check Version Numbers
	↓
Clear Cache & Hard Refresh
	↓
Restart Server
	↓
Check Recent Changes (git log)
	↓
If persists: Check GitHub Issues
	↓
Create Issue Report
```

### Issue Report Template

```markdown
## Issue: [Description]

### Environment
- Browser: Chrome/Firefox/Edge/Safari
- Version: 
- OS: Windows/Mac/Linux
- Node Version: 
- Python Version: 

### Steps to Reproduce
1. 
2. 
3. 

### Expected Behavior


### Actual Behavior


### Screenshots


### Error Messages


### Additional Context

```

## 📋 Rollback Plan

If issues occur with new version:

```powershell
# Revert to previous version
git revert HEAD

# Reinstall dependencies
npm install

# Restart server
dredge interactive --port 8000

# Verify
Start-Process "http://127.0.0.1:8000"
```

## ✅ Post-Deployment Verification

After deployment, verify:

- [ ] Credits footer visible
- [ ] No console errors
- [ ] No deprecation warnings
- [ ] GitHub link works
- [ ] Responsive on mobile
- [ ] Performance acceptable
- [ ] All team members notified
- [ ] Documentation updated

## 📞 Contact

For issues or questions:
- **Repository**: https://github.com/QueenFi703/DREDGE-Cli
- **Creator**: QueenFi703
- **Maintainer**: Dredge Agent
- **Security**: QueenFi703 & Dredge Agent

---

**Deployment & Maintenance Guide Complete** ✅
