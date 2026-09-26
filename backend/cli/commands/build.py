import typer
from pathlib import Path
from ..utils.console import console, print_header
from ..utils.dependency import get_intelligence_core
from core.self_coder import SelfCoder

app = typer.Typer(help="Self-coding and agent generation module")

@app.command("build")
def build(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Description of the agent or module to build"),
    name: str = typer.Option(None, "--name", "-n", help="Optional exact filename (e.g., data_scientist.py)")
):
    """
    Use Syntera's intelligence to design, build, and test a new module automatically.
    """
    print_header("Self-Coder Engine")
    
    console.print(f"[info]Requirements:[/info] {prompt}")
    if name:
        console.print(f"[info]Target Name:[/info] {name}")
        
    core = get_intelligence_core()
    coder = SelfCoder(core)
    
    with console.status("[bold green]Applying Fabric patterns and generating code via Nemotron-3-Ultra...") as status:
        try:
            file_path, test_path, msg = coder.generate_agent(prompt, name)
            
            console.print(f"[success]✓ Generated[/success] {file_path.relative_to(Path.cwd())}")
            console.print(f"[success]✓ Generated[/success] {test_path.relative_to(Path.cwd())}")
            
            status.update("[bold cyan]Running generated test suite...")
            success, output = coder.run_tests(test_path)
            
            passed = output.count("PASSED")
            failed = output.count("FAILED")
            total = passed + failed
            
            if success:
                console.print(f"[success]✓ All tests passed ({passed}/{total})[/success]")
            else:
                console.print(f"[warning]⚠ Some tests failed ({failed} failed, {passed} passed)[/warning]")
                
            console.print(f"\n[bold green]Ready to review at:[/bold green] {file_path.relative_to(Path.cwd())}")
            
        except Exception as e:
            console.print(f"[danger]Build failed:[/danger] {str(e)}")
            raise typer.Exit(code=1)
