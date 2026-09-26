import typer
from typing import Optional
import os
from pathlib import Path
import json
from ..utils.console import console, print_header

app = typer.Typer(help="Authentication and cloud connection management")

CONFIG_DIR = Path.home() / ".syntera"
CRED_FILE = CONFIG_DIR / "credentials.json"

@app.command("login")
def login(api_key: Optional[str] = typer.Option(None, "--key", "-k", help="NVIDIA API Key")):
    """
    Authenticate your CLI session with cloud services.
    """
    print_header("Authentication")
    
    if not api_key:
        api_key = typer.prompt("Enter your NVIDIA API Key (hidden)", hide_input=True)
        
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Simple local storage for MVP (Day 1 structure)
    creds = {"nvidia_api_key": api_key, "tier": "professional"}
    with open(CRED_FILE, "w") as f:
        json.dump(creds, f)
        
    # We also attempt to sync it to local .env in the root if it's running from source
    root_env = Path.cwd() / ".env"
    env_content = f"NVIDIA_API_KEY={api_key}\n"
    with open(root_env, "w") as f:
        f.write(env_content)
        
    console.print("[success]✓ Successfully authenticated and credentials saved locally![/success]")
    console.print("[info]Your CLI will now route complex tasks to remote frontier models.[/info]")

@app.command("status")
def status():
    """Check current authentication status and tier."""
    print_header("Status")
    if CRED_FILE.exists():
        with open(CRED_FILE, "r") as f:
            creds = json.load(f)
        console.print(f"[success]Authenticated.[/success] Current Tier: [bold cyan]{creds.get('tier', 'Unknown')}[/bold cyan]")
    else:
        console.print("[warning]Not authenticated. Running in local-only open-source mode.[/warning]")
