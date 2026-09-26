import typer
from pathlib import Path
from ..utils.console import console, print_header
from ..utils.dependency import get_intelligence_core
from intelligence.contracts import TaskComplexity

app = typer.Typer(help="Code quality and review agents")

@app.command("review")
def review(
    target: Path = typer.Argument(..., help="Path to the code file or directory to review"),
    strictness: str = typer.Option("normal", "--strictness", "-s", help="Review strictness (relaxed, normal, pedantic)")
):
    """
    Automated code review across security, performance, and best practices.
    """
    print_header("Code Quality Agent")
    
    if not target.exists():
        console.print(f"[danger]Error: Target {target} does not exist.[/danger]")
        raise typer.Exit(code=1)
        
    console.print(f"[info]Scanning:[/info] {target}")
    console.print(f"[info]Strictness Level:[/info] {strictness}")
    
    # Boot the core
    core = get_intelligence_core()
    
    try:
        from agents.code_quality import CodeQualityAgent
        
        with console.status("[bold green]Analyzing code structure and smells...") as status:
            cq_agent = CodeQualityAgent()
            result = cq_agent.review(str(target), strictness)
            
        if 'error' in result:
            console.print(f"[danger]Review Failed:[/danger] {result['error']}")
        else:
            console.print("\n[bold cyan]Code Review Results:[/bold cyan]")
            
            issues = result.get('issues', [])
            if not issues:
                console.print("[success]✓ No issues found! Code is perfectly clean.[/success]")
            else:
                for issue in issues:
                    color = "danger" if issue['severity'] == "high" else "warning"
                    console.print(f"[{color}]• [{issue['severity'].upper()}] {issue['type'].upper()}: {issue['message']}[/{color}]")
                    
            console.print(f"\n[bold white]Refactoring Suggestion:[/bold white] {result.get('refactoring')}")
            console.print(f"\n[dim]Executed by: CodeQualityAgent (Local Compute)[/dim]")
            
    except ImportError:
        console.print("[danger]CodeQualityAgent module not found or incomplete.[/danger]")
        raise typer.Exit(code=1)
