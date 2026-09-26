from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="syntera-core",
    version="0.1.0",
    description="Syntera - Hybrid Open-Source + Premium AI Agentic Engine",
    author="Syntera Team",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "syntera=backend.cli.main:app",
        ],
    },
    python_requires=">=3.11",
)
