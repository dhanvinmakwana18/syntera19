# Syntera - AI Agent for Data Science & Code Quality

Syntera is a privacy-first, hardware-aware AI orchestration and agentic workflow engine. It automatically routes between your local edge hardware and remote cloud endpoints to execute heavy agentic workflows like end-to-end Data Science modeling and automated Code Review.

## 🚀 Installation

Install the Syntera Core CLI package directly from pip:

```bash
pip install syntera-core
```

## ⚙️ Setup Instructions

To unlock remote reasoning capabilities via the NVIDIA ecosystem, simply add your API key:

```bash
# Set your API key via the CLI
syntera login --key <your_nvidia_api_key>

# Alternatively, copy the environment template
cp .env.example .env
```
Ensure you update `.env` with your `NVIDIA_API_KEY=your_key_here`.

## 🛠️ Quick Start Examples

Syntera is built entirely around specialized Premium Agents that automate entire workflows.

**1. Data Scientist Agent:**
Perform Exploratory Data Analysis, Feature Engineering, and train 5 models automatically from a single dataset.
```bash
syntera analyze dataset.csv --agent data-scientist
```

**2. Data Scientist Web UI (Streamlit):**
Launch the interactive web frontend for the Data Science Agent to upload multiple CSVs and generate Enterprise PDF reports dynamically.
```bash
streamlit run ds_frontend.py
```

**3. Code Quality Agent:**
Automate code review across security risks, performance issues, and PEP-8 styling standards.
```bash
syntera review my_script.py --strictness pedantic
```

## 💻 Development

Want to test or modify Syntera locally? 

```bash
# Clone the repository
git clone https://github.com/yourusername/syntera19.git
cd syntera19

# Install dependencies
pip install -r requirements.txt

# Run the test suite
pytest tests/ -v

# Run the CLI locally
.\syntera.cmd --help
```
