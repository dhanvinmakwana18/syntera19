from rich.console import Console
from rich.theme import Theme

syntera_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "success": "bold green",
    "prompt": "bold blue"
})

console = Console(theme=syntera_theme)

def print_header(title: str):
    console.print(f"[bold cyan]Syntera[/bold cyan] » [bold white]{title}[/bold white]")
