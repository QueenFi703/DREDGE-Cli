#!/usr/bin/env powershell
<#
.SYNOPSIS
	Swift Package Manager dependency resolution for DREDGE
.DESCRIPTION
	Resolves DREDGE Swift dependencies using Docker dev image
.PARAMETER Command
	Operation: resolve, build, test, clean, shell
.PARAMETER Detailed
	Show detailed output
#>

param(
	[Parameter(Mandatory=$true)]
	[ValidateSet('resolve', 'build', 'describe', 'test', 'clean', 'shell', 'run')]
	[string]$Command,

	[switch]$Detailed
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$imageName = "dredge-dev:latest"
$volumeArgs = @("-v", "${projectRoot}:/workspace")

# Colors
$Green = "`e[32m"
$Yellow = "`e[33m"
$Cyan = "`e[36m"
$Red = "`e[31m"
$Reset = "`e[0m"

function Write-Status { param([string]$Message); Write-Host "$Green► $Message$Reset" }
function Write-Info { param([string]$Message); Write-Host "$Cyan► $Message$Reset" }
function Write-Error-Custom { param([string]$Message); Write-Host "$Red✗ $Message$Reset" }

try {
	Set-Location $projectRoot

	switch($Command) {
		'resolve' {
			Write-Status "Resolving DREDGE Swift dependencies..."

			# Check if image exists
			$imageExists = docker images --quiet $imageName
			if (-not $imageExists) {
				Write-Error-Custom "Docker image '$imageName' not found!"
				Write-Info "Build it with: docker buildx build --platform linux/amd64 --target dev -t dredge-dev:latest --load ."
				exit 1
			}

			Write-Info "Running: swift package update"
			docker run --rm $volumeArgs $imageName `
				bash -c "cd /workspace && swift package update"

			if ($LASTEXITCODE -eq 0) {
				Write-Status "✓ Dependencies resolved!"
				Write-Info "Package.resolved created"
			} else {
				Write-Error-Custom "Resolution failed with exit code $LASTEXITCODE"
				exit 1
			}
		}

		'describe' {
			Write-Status "Swift package information..."
			docker run --rm $volumeArgs $imageName `
				bash -c "cd /workspace && swift package describe"
		}

		'build' {
			Write-Status "Building DREDGE..."
			Write-Info "This may take 2-5 minutes..."

			docker run --rm $volumeArgs $imageName `
				bash -c "cd /workspace && swift package update && swift build"

			if ($LASTEXITCODE -eq 0) {
				Write-Status "Build complete!"
				Write-Info "Executable: .build/debug/dredge-cli"
			}
		}

		'test' {
			Write-Status "Running Swift tests..."
			docker run --rm $volumeArgs $imageName `
				bash -c "cd /workspace && swift test"
		}

		'clean' {
			Write-Status "Cleaning build artifacts..."
			Remove-Item -Path ".build" -Recurse -Force -ErrorAction SilentlyContinue
			Remove-Item -Path "Package.resolved" -Force -ErrorAction SilentlyContinue
			Write-Status "Clean complete"
		}

		'shell' {
			Write-Status "Opening development shell..."
			Write-Info "You can now run Swift commands directly"
			Write-Info "Example: swift package update && swift build"

			docker run -it $volumeArgs $imageName /bin/bash
		}

		'run' {
			Write-Status "Running dredge-cli..."
			docker run --rm $volumeArgs $imageName `
				bash -c "cd /workspace/.build/debug && ./dredge-cli --version"
		}
	}
}
catch {
	Write-Error-Custom "Error: $_"
	exit 1
}

Write-Host ""
