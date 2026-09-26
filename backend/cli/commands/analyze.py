import typer
from pathlib import Path
from ..utils.console import console, print_header
from ..utils.dependency import get_intelligence_core
from intelligence.contracts import TaskComplexity

app = typer.Typer(help="Data and file analysis agents")

@app.command("analyze")
def analyze(
    file_path: Path = typer.Argument(..., help="Path to the file or dataset to analyze"),
    agent: str = typer.Option("data-scientist", "--agent", "-a", help="Which premium agent to use"),
    goal: str = typer.Option(None, "--goal", "-g", help="Specific goal for the analysis")
):
    """
    Analyze a dataset or document using specialized agents (e.g. Data Scientist Agent).
    """
    print_header(f"{agent.replace('-', ' ').title()} Analysis")
    
    if not file_path.exists():
        console.print(f"[danger]Error: File {file_path} does not exist.[/danger]")
        raise typer.Exit(code=1)
        
    console.print(f"[info]Loading file:[/info] {file_path.name}")
    console.print(f"[info]Target Agent:[/info] {agent}")
    if goal:
        console.print(f"[info]Analysis Goal:[/info] {goal}")
        
    # Boot the core
    core = get_intelligence_core()
    
    try:
        from agents.data_scientist import DataScientistAgent
        
        with console.status(f"[bold green]Agent {agent} is crunching numbers...") as status:
            ds_agent = DataScientistAgent()
            result = ds_agent.analyze(str(file_path), goal)
            
        if 'error' in result:
            console.print(f"[danger]Analysis Failed:[/danger] {result['error']}")
        else:
            console.print("\n[bold cyan]Analysis Complete:[/bold cyan]")
            console.print(f"[bold white]Rows Analyzed:[/bold white] {result.get('eda_summary', {}).get('rows')}")
            console.print(f"[bold white]Best Model:[/bold white] {result.get('best_model', {}).get('name')} (Acc: {result.get('best_model', {}).get('accuracy')})")
            console.print(f"\n[bold white]Deployment Snippet:[/bold white]\n```python\n{result.get('deployment_code')}\n```")
            console.print(f"\n[dim]Executed by: DataScientistAgent (Local Compute)[/dim]")
            
    except ImportError:
        console.print("[danger]DataScientistAgent module not found or incomplete.[/danger]")
        raise typer.Exit(code=1)
