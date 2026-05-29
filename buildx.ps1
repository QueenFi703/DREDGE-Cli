#!/usr/bin/env pwsh
<#
.SYNOPSIS
	Docker Buildx wrapper for DREDGE CI/CD operations
.DESCRIPTION
	Simplifies common Docker Buildx commands with pre-configured settings for DREDGE
.PARAMETER Command
	Operation: build, bake, local, push, clean, inspect
.PARAMETER Target
	Docker target: prod, dev, gpu, cpu, all
.PARAMETER Registry
	Registry: docker.io, ghcr.io, acr (Azure Container Registry)
.PARAMETER Version
	Version tag (default: latest)
.PARAMETER Push
	Push to registry after build
#>

param(
	[Parameter(Mandatory=$true)]
	[ValidateSet('build', 'bake', 'local', 'push', 'clean', 'inspect')]
	[string]$Command,

	[ValidateSet('prod', 'dev', 'gpu', 'cpu', 'all', 'base')]
	[string]$Target = 'prod',

	[ValidateSet('docker.io', 'ghcr.io', 'acr')]
	[string]$Registry = 'docker.io',

	[string]$Username = $env:DOCKER_USERNAME,

	[string]$Version = 'latest',

	[switch]$Push,

	[string]$Namespace = 'dredge'
)

# Colors for output
$Green = "`e[32m"
$Yellow = "`e[33m"
$Red = "`e[31m"
$Reset = "`e[0m"

function Write-Status {
	param([string]$Message, [string]$Color = $Green)
	Write-Host "$Color► $Message$Reset"
}

function Write-Error-Custom {
	param([string]$Message)
	Write-Host "$Red✗ $Message$Reset"
}

# Determine registry URL
$RegistryUrl = switch($Registry) {
	'docker.io' { 'docker.io' }
	'ghcr.io' { 'ghcr.io' }
	'acr' { "myregistry.azurecr.io" }
}

$ImageBase = "$RegistryUrl/$Namespace"

# Map targets to build config
$TargetMap = @{
	'prod' = @{
		target   = 'prod'
		platform = 'linux/amd64,linux/arm64'
		tags     = @("$ImageBase/dredge:$Version", "$ImageBase/dredge:latest")
	}
	'dev' = @{
		target   = 'dev'
		platform = 'linux/amd64,linux/arm64'
		tags     = @("$ImageBase/dredge-dev:$Version", "$ImageBase/dredge-dev:latest")
	}
	'gpu' = @{
		target   = 'gpu-build'
		platform = 'linux/amd64'
		tags     = @("$ImageBase/dredge-gpu:$Version", "$ImageBase/dredge-gpu:latest")
	}
	'cpu' = @{
		target   = 'cpu-build'
		platform = 'linux/amd64,linux/arm64'
		tags     = @("$ImageBase/dredge-cpu:$Version", "$ImageBase/dredge-cpu:latest")
	}
	'base' = @{
		target   = 'base'
		platform = 'linux/amd64,linux/arm64'
		tags     = @("$ImageBase/dredge-base:$Version", "$ImageBase/dredge-base:latest")
	}
}

# Execution logic
try {
	switch($Command) {
		'local' {
			# Local single-platform build (load into Docker)
			Write-Status "Building $Target locally (linux/amd64, load to Docker)..."

			$config = $TargetMap[$Target]
			$tags = $config.tags | ForEach-Object { "--tag $_" } | Join-String -Separator ' '

			$cmd = "docker buildx build --platform linux/amd64 --target $($config.target) $tags --load ."

			Write-Host "$Yellow$cmd$Reset`n"
			Invoke-Expression $cmd

			if ($LASTEXITCODE -eq 0) {
				Write-Status "Build successful! Run with: docker run $($config.tags[0])"
			} else {
				Write-Error-Custom "Build failed with exit code $LASTEXITCODE"
			}
		}

		'build' {
			# Multi-platform build with provenance & SBOM
			Write-Status "Building $Target (multi-platform with provenance & SBOM)..."

			if (-not $Push) {
				Write-Status "Use --Push to push to registry"
			}

			$config = $TargetMap[$Target]
			$tags = $config.tags | ForEach-Object { "--tag $_" } | Join-String -Separator ' '
			$pushFlag = if ($Push) { "--push" } else { "--output type=oci" }

			$cmd = "docker buildx build `
				--platform $($config.platform) `
				--target $($config.target) `
				--provenance=mode=max `
				--sbom=true `
				$tags `
				$pushFlag `
				."

			Write-Host "$Yellow$cmd$Reset`n"
			Invoke-Expression $cmd

			if ($LASTEXITCODE -eq 0) {
				Write-Status "Build successful!"
			}
		}

		'bake' {
			# Use docker-bake.hcl configuration
			Write-Status "Running docker buildx bake for $Target..."

			$pushFlag = if ($Push) { "--push" } else { "" }

			$cmd = "docker buildx bake -f docker-bake.hcl `
				--set '*.args.REGISTRY=$RegistryUrl' `
				--set '*.args.VERSION=$Version' `
				$pushFlag `
				$Target"

			Write-Host "$Yellow$cmd$Reset`n"
			Invoke-Expression $cmd

			if ($LASTEXITCODE -eq 0) {
				Write-Status "Bake successful!"
			}
		}

		'push' {
			# Login and push all tags
			Write-Status "Preparing to push to $Registry..."

			# Login logic
			switch($Registry) {
				'docker.io' {
					if (-not $Username) {
						Write-Error-Custom "Provide DOCKER_USERNAME environment variable or --Username parameter"
						exit 1
					}
					Write-Status "Logging in to Docker Hub as $Username..."
					Write-Host "Password prompt will appear..."
					docker login -u $Username
				}
				'ghcr.io' {
					Write-Status "Logging in to GitHub Container Registry..."
					Write-Host "Use GitHub Personal Access Token as password"
					docker login ghcr.io -u $Username
				}
				'acr' {
					Write-Status "Logging in to Azure Container Registry..."
					az acr login --name myregistry
				}
			}

			if ($LASTEXITCODE -eq 0) {
				Write-Status "Login successful! Now run: buildx.ps1 build -Target $Target -Push"
			}
		}

		'inspect' {
			# Show builder and build info
			Write-Status "Docker Buildx Status:"
			Write-Host "`n--- Builders ---"
			docker buildx ls

			Write-Host "`n--- Builder Details ---"
			docker buildx inspect

			Write-Host "`n--- Disk Usage ---"
			docker buildx du

			Write-Host "`n--- Image Tags for $Target ---"
			$TargetMap[$Target].tags | ForEach-Object { Write-Host "  $_" }
		}

		'clean' {
			# Clean builder cache
			Write-Status "Cleaning Docker Buildx builder..."
			docker buildx prune -f
			Write-Status "Builder cache cleaned"
		}
	}
}
catch {
	Write-Error-Custom "Error: $_"
	exit 1
}

Write-Host ""
