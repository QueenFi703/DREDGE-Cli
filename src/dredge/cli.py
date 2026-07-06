"""
DREDGE CLI System - Complete Command-Line Interface

Based on phone screenshots showing:
7. TELEMETRY LAYER
8. DAG DEFINITION (Real Pipeline Graph)
9. CLI SYSTEM
10. CLI USAGE

Implements the full CLI for DREDGE architecture interaction.
"""

import asyncio
import json
import sys
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import click
from datetime import datetime

logger = logging.getLogger(__name__)


class CLIMode(Enum):
    """CLI execution modes"""
    INTERACTIVE = "interactive"
    PIPELINE = "pipeline"
    TRANSLATE = "translate"
    ANALYZE = "analyze"
    STATUS = "status"
    CONFIG = "config"


@dataclass
class CLIConfig:
    """CLI configuration"""
    verbose: bool = False
    json_output: bool = False
    cache_enabled: bool = True
    pipeline_type: str = "standard"
    timeout: int = 30
    log_level: str = "INFO"
    output_format: str = "table"  # table, json, csv


class CLIFormatter:
    """Format output for different modes"""

    @staticmethod
    def format_table(data: Dict[str, Any], title: str = "") -> str:
        """Format as ASCII table"""
        lines = []
        
        if title:
            lines.append(f"\n{'═' * 60}")
            lines.append(f"  {title}")
            lines.append(f"{'═' * 60}\n")
        
        if isinstance(data, dict):
            for key, value in data.items():
                lines.append(f"  {key:<30} {str(value)}")
        
        lines.append(f"\n{'═' * 60}\n")
        return "\n".join(lines)

    @staticmethod
    def format_json(data: Dict[str, Any]) -> str:
        """Format as JSON"""
        return json.dumps(data, indent=2)

    @staticmethod
    def format_telemetry(telemetry: Dict[str, Any]) -> str:
        """Format telemetry data"""
        lines = []
        lines.append("\n┌─ TELEMETRY LAYER ─────────────────────────────┐")
        
        if "event_count" in telemetry:
            lines.append(f"│ Events: {telemetry['event_count']}")
        
        if "metrics" in telemetry:
            lines.append(f"│ Metrics:")
            for name, value in telemetry["metrics"].items():
                lines.append(f"│   • {name}: {value}")
        
        lines.append("└────────────────────────────────────────────────┘\n")
        return "\n".join(lines)

    @staticmethod
    def format_dag_graph(results: Dict[str, Any]) -> str:
        """Format DAG execution graph"""
        lines = []
        lines.append("\n┌─ DAG DEFINITION (REAL PIPELINE GRAPH) ─────┐")
        
        node_results = results.get("results", {})
        for i, (node_id, result) in enumerate(node_results.items(), 1):
            lines.append(f"│ {i}. {node_id.upper()}")
            if isinstance(result, dict):
                for key, val in result.items():
                    val_str = json.dumps(val) if not isinstance(val, str) else val
                    lines.append(f"│    └─ {key}: {val_str[:40]}")
        
        lines.append("└────────────────────────────────────────────┘\n")
        return "\n".join(lines)


class DREDGECLI:
    """Main CLI interface for DREDGE"""

    def __init__(self, config: Optional[CLIConfig] = None):
        self.config = config or CLIConfig()
        self.formatter = CLIFormatter()
        self.setup_logging()

    def setup_logging(self):
        """Setup logging"""
        level = getattr(logging, self.config.log_level, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    async def run_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a pipeline"""
        from .architecture import dredge_run_pipeline

        click.echo(click.style("▶ Executing DREDGE Pipeline...", fg="cyan", bold=True))
        
        try:
            result = await dredge_run_pipeline(
                input_data,
                pipeline_type=self.config.pipeline_type
            )
            
            click.echo(click.style("✓ Pipeline completed", fg="green"))
            return result
        
        except Exception as e:
            click.echo(click.style(f"✗ Pipeline failed: {e}", fg="red"), err=True)
            raise

    async def run_translate(self, text: str, src: str, tgt: str) -> Dict[str, Any]:
        """Execute translation"""
        from .providers import execute_translation_chain

        click.echo(click.style(f"▶ Translating: {src} → {tgt}...", fg="cyan"))
        
        try:
            result = await execute_translation_chain({
                "text": text,
                "source_language": src,
                "target_language": tgt
            })
            
            click.echo(click.style("✓ Translation complete", fg="green"))
            return result
        
        except Exception as e:
            click.echo(click.style(f"✗ Translation failed: {e}", fg="red"), err=True)
            raise

    async def run_analyze(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute analysis"""
        from .providers import execute_analysis_chain

        click.echo(click.style(f"▶ Analyzing: {query}...", fg="cyan"))
        
        try:
            result = await execute_analysis_chain({
                "query": query,
                "context": context or {}
            })
            
            click.echo(click.style("✓ Analysis complete", fg="green"))
            return result
        
        except Exception as e:
            click.echo(click.style(f"✗ Analysis failed: {e}", fg="red"), err=True)
            raise

    def format_output(self, data: Dict[str, Any]) -> str:
        """Format output based on config"""
        if self.config.output_format == "json":
            return self.formatter.format_json(data)
        elif self.config.output_format == "table":
            output = []
            
            # Format main result
            if "result" in data:
                output.append(self.formatter.format_table(
                    data.get("result", {}),
                    "RESULT"
                ))
            
            # Format DAG graph
            if "results" in data:
                output.append(self.formatter.format_dag_graph(data))
            
            # Format telemetry
            if "telemetry" in data:
                output.append(self.formatter.format_telemetry(data["telemetry"]))
            
            return "\n".join(output)
        
        return str(data)

    def print_output(self, data: Dict[str, Any]):
        """Print formatted output"""
        output = self.format_output(data)
        click.echo(output)


# ============================================================================
# CLICK CLI COMMANDS
# ============================================================================

@click.group(invoke_without_command=True)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--json", "-j", "json_output", is_flag=True, help="JSON output")
@click.option("--config-file", "-c", type=click.Path(), help="Config file")
@click.pass_context
def cli(ctx, verbose, json_output, config_file):
    """
    DREDGE CLI - Advanced AI Pipeline System
    
    Unified interface for:
    - Pipeline execution
    - Translation & analysis
    - Provider management
    - System monitoring
    """
    config = CLIConfig(
        verbose=verbose,
        json_output=json_output,
        output_format="json" if json_output else "table"
    )
    
    cli_instance = DREDGECLI(config)
    ctx.ensure_object(dict)
    ctx.obj["cli"] = cli_instance
    ctx.obj["config"] = config


@cli.command()
@click.argument("input_file", type=click.Path(exists=True), required=False)
@click.option("--pipeline-type", "-p", default="standard", 
              help="Pipeline type (standard or ios_swift)")
@click.option("--query", "-q", help="Query string")
@click.pass_context
def pipeline(ctx, input_file, pipeline_type, query):
    """
    Execute a DREDGE pipeline
    
    Examples:
        dredge pipeline input.json
        dredge pipeline --query "test query"
    """
    cli_instance = ctx.obj["cli"]
    cli_instance.config.pipeline_type = pipeline_type
    
    if input_file:
        with open(input_file) as f:
            input_data = json.load(f)
    elif query:
        input_data = {"query": query}
    else:
        click.echo("Error: Provide --query or input file", err=True)
        sys.exit(1)
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(cli_instance.run_pipeline(input_data))
    cli_instance.print_output(result)


@cli.command()
@click.argument("text")
@click.option("--source", "-s", default="en", help="Source language")
@click.option("--target", "-t", default="es", help="Target language")
@click.pass_context
def translate(ctx, text, source, target):
    """
    Translate text using DREDGE providers
    
    Examples:
        dredge translate "Hello" --source en --target es
        dredge translate "Bonjour" -s fr -t en
    """
    cli_instance = ctx.obj["cli"]
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(
        cli_instance.run_translate(text, source, target)
    )
    cli_instance.print_output(result)


@cli.command()
@click.argument("query")
@click.option("--context", "-c", type=click.Path(exists=True), 
              help="Context file (JSON)")
@click.pass_context
def analyze(ctx, query, context):
    """
    Analyze text using DREDGE semantic analysis
    
    Examples:
        dredge analyze "What is quantum computing?"
        dredge analyze "Explain AI" --context context.json
    """
    cli_instance = ctx.obj["cli"]
    
    context_data = {}
    if context:
        with open(context) as f:
            context_data = json.load(f)
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(
        cli_instance.run_analyze(query, context_data)
    )
    cli_instance.print_output(result)


@cli.command()
@click.pass_context
def status(ctx):
    """
    Get DREDGE system status
    
    Shows:
    - Provider health
    - Cache status
    - Pipeline availability
    - System metrics
    """
    from .providers import get_provider_status
    
    click.echo(click.style("▶ Checking system status...", fg="cyan"))
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    status = loop.run_until_complete(get_provider_status())
    
    cli_instance = ctx.obj["cli"]
    cli_instance.print_output({"providers_status": status})


@cli.command()
@click.option("--cache/--no-cache", default=True, help="Enable caching")
@click.option("--pipeline-type", "-p", default="standard", help="Default pipeline type")
@click.option("--log-level", "-l", default="INFO", help="Log level")
@click.pass_context
def config(ctx, cache, pipeline_type, log_level):
    """
    Configure DREDGE CLI
    
    Examples:
        dredge config --no-cache
        dredge config --pipeline-type ios_swift
        dredge config --log-level DEBUG
    """
    cli_instance = ctx.obj["cli"]
    cli_instance.config.cache_enabled = cache
    cli_instance.config.pipeline_type = pipeline_type
    cli_instance.config.log_level = log_level
    
    output = {
        "cache_enabled": cache,
        "pipeline_type": pipeline_type,
        "log_level": log_level,
        "status": "configured"
    }
    
    click.echo(click.style("Configuration updated:", fg="green", bold=True))
    cli_instance.print_output(output)


@cli.command()
@click.pass_context
def interactive(ctx):
    """
    Interactive DREDGE CLI session
    
    Commands:
        help              Show help
        pipeline          Run pipeline
        translate         Translate text
        analyze          Analyze query
        status           System status
        config           Show configuration
        exit             Exit interactive mode
    """
    click.echo(click.style("\n╔════════════════════════════════════════╗", fg="cyan"))
    click.echo(click.style("║    DREDGE Interactive CLI v1.0         ║", fg="cyan"))
    click.echo(click.style("║    Type 'help' for commands            ║", fg="cyan"))
    click.echo(click.style("╚════════════════════════════════════════╝\n", fg="cyan"))
    
    cli_instance = ctx.obj["cli"]
    
    while True:
        try:
            user_input = click.prompt(click.style("dredge", fg="cyan", bold=True))
            
            if user_input.lower() == "exit":
                click.echo(click.style("Goodbye!", fg="cyan"))
                break
            
            elif user_input.lower() == "help":
                click.echo("""
Commands:
  pipeline [query]       - Execute pipeline with query
  translate [text]       - Translate text
  analyze [query]        - Analyze query
  status                 - Show system status
  config                 - Show configuration
  help                   - Show this help
  exit                   - Exit interactive mode
                """)
            
            elif user_input.lower().startswith("pipeline"):
                query = user_input.replace("pipeline", "").strip()
                if query:
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    result = loop.run_until_complete(
                        cli_instance.run_pipeline({"query": query})
                    )
                    cli_instance.print_output(result)
            
            elif user_input.lower().startswith("translate"):
                text = user_input.replace("translate", "").strip()
                if text:
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    result = loop.run_until_complete(
                        cli_instance.run_translate(text, "en", "es")
                    )
                    cli_instance.print_output(result)
            
            elif user_input.lower() == "status":
                from .providers import get_provider_status
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                status = loop.run_until_complete(get_provider_status())
                cli_instance.print_output({"providers_status": status})
        
        except KeyboardInterrupt:
            click.echo(click.style("\nInterrupted. Type 'exit' to quit.", fg="yellow"))
        except Exception as e:
            click.echo(click.style(f"Error: {e}", fg="red"), err=True)


@cli.command()
@click.pass_context
def version(ctx):
    """Show DREDGE version"""
    from . import __version__
    click.echo(f"DREDGE CLI v{__version__}")


# ============================================================================
# USAGE GUIDE
# ============================================================================

USAGE_GUIDE = """
╔═══════════════════════════════════════════════════════════════════════╗
║                    DREDGE CLI USAGE GUIDE                            ║
╚═══════════════════════════════════════════════════════════════════════╝

INSTALLATION:
  pip install dredge-cli

BASIC COMMANDS:
  dredge --help              Show help
  dredge --version           Show version
  dredge pipeline -q "query" Execute pipeline
  dredge translate "text"    Translate text
  dredge analyze "question"  Analyze query
  dredge status              Show system status

OPTIONS:
  -v, --verbose              Verbose output
  -j, --json                 JSON output format
  -p, --pipeline-type TYPE   Pipeline type (standard|ios_swift)
  -s, --source LANG          Source language (for translate)
  -t, --target LANG          Target language (for translate)

EXAMPLES:

1. Pipeline Execution:
   dredge pipeline -q "What is quantum computing?"
   dredge pipeline input.json --pipeline-type standard

2. Translation:
   dredge translate "Hello world" -s en -t es
   dredge translate "Bonjour" -s fr -t en

3. Analysis:
   dredge analyze "Explain machine learning"
   dredge analyze "What is AI?" --context context.json

4. Status Monitoring:
   dredge status
   dredge status --json

5. Configuration:
   dredge config --no-cache
   dredge config --log-level DEBUG

6. Interactive Mode:
   dredge interactive
   > pipeline test query
   > translate Hello
   > status
   > exit

OUTPUT FORMATS:
  Table (default):
    Formatted ASCII tables with results
    
  JSON (-j):
    Full JSON response for scripting
    
  Verbose (-v):
    Detailed execution trace

CONFIGURATION FILE:
  Create ~/.dredge/config.json:
  {
    "cache_enabled": true,
    "pipeline_type": "standard",
    "log_level": "INFO",
    "output_format": "table"
  }

ENVIRONMENT VARIABLES:
  DREDGE_PIPELINE_TYPE    Default pipeline type
  DREDGE_LOG_LEVEL        Default log level
  DREDGE_CACHE_ENABLED    Enable/disable caching
  DREDGE_OUTPUT_FORMAT    Output format (table|json|csv)

TROUBLESHOOTING:
  - No output: Use --json to see full response
  - Slow execution: Check --cache setting
  - Missing providers: Run 'dredge status' to verify
  - Debug info: Use -v (verbose) flag

FOR MORE HELP:
  dredge <command> --help
  dredge interactive
"""


def print_usage():
    """Print usage guide"""
    click.echo(USAGE_GUIDE)


if __name__ == "__main__":
    cli()
