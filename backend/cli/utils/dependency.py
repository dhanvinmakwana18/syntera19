import os
import sys
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn
from .console import console

def get_intelligence_core():
    # Add backend to path dynamically if needed
    backend_dir = Path(__file__).parent.parent.parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
        
    # Load environment variables
    from dotenv import load_dotenv
    root_env = backend_dir.parent / '.env'
    load_dotenv(root_env)
    
    from core.container import build_container
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
        progress.add_task(description="Initializing Syntera Intelligence Engine...", total=None)
        container = build_container()
        return container.get_intelligence()
