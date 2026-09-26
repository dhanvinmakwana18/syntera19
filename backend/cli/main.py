import typer
from .commands import auth, analyze, review, build

app = typer.Typer(
    name="syntera",
    help="Syntera: Hybrid Open-Source + Premium Agents Engine",
    no_args_is_help=True
)

# Mount command domains
app.add_typer(auth.app, name="auth", help="Authentication and credentials")
# For better UX, we'll expose analyze, review, and build directly at the top level
app.command(name="analyze", help=analyze.analyze.__doc__)(analyze.analyze)
app.command(name="review", help=review.review.__doc__)(review.review)
app.command(name="build", help=build.build.__doc__)(build.build)

# Fallback top-level commands pointing to the submodules
app.command(name="login", help="Authenticate your CLI session")(auth.login)
app.command(name="status", help="Check current authentication status")(auth.status)


@app.callback()
def main():
    """
    Syntera Autonomous AI Agent Platform
    """
    pass

if __name__ == "__main__":
    app()
