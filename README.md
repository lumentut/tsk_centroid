# tsk_centroid

A Centroid-Based TSK Fuzzy Logic System
This project uses Python virtual environments to manage dependencies and ensure consistent development environments across different machines.

## Prerequisites

- Python 3.6 or higher installed on your system
- `pip` (Python package installer)

## Setting Up the Virtual Environment

### 1. Create a Virtual Environment

```bash
# Create a new virtual environment named 'venv'
python -m venv venv

# On some systems, you might need to use python3
python3 -m venv venv
```

### 2. Activate the Virtual Environment

**On Windows:**

```bash
# Command Prompt
venv\Scripts\activate

# PowerShell
venv\Scripts\Activate.ps1

# Git Bash
source venv/Scripts/activate
```

**On macOS/Linux:**

```bash
source venv/bin/activate
```

When activated, you should see `(venv)` at the beginning of your command prompt.

### 3. Install Dependencies

```bash
# Upgrade pip to the latest version
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### 4. Deactivate the Virtual Environment

When you're done working on the project:

```bash
deactivate
```

## Managing Dependencies

### Adding New Dependencies

1. Make sure your virtual environment is activated
2. Install the package:
   ```bash
   pip install package-name
   ```
3. Update the requirements file:
   ```bash
   pip freeze > requirements.txt
   ```

### Updating Dependencies

```bash
# Update a specific package
pip install --upgrade package-name

# Update requirements.txt after changes
pip freeze > requirements.txt
```

## Project Structure

```
tsk_centroid/
├── .vscode/              # VS Code configs
├── notebooks/            # Jupyter Notebook playground
├── src/                  # Source code
├── venv/                 # Virtual environment (ignored by Git)
├── .gitignore           # Git ignore file
├── LICENSE              # LICENSE file
├── README.md            # This file
├── EXPERIMENT_GUIDE.md   # Detailed experiment instructions
└── requirements.txt      # Project dependencies
```

## Running Experiments

This project contains multiple Jupyter notebook experiments for IT2TSK fuzzy inference system research. For detailed instructions on how to run each experiment, including:

- Step-by-step experiment workflow (Experiments 1-6)
- Parameter configuration and optimization
- Expected outputs and execution times
- Troubleshooting common issues
- Performance evaluation guidelines

**👉 See [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) for complete experiment instructions.**

## Important Notes

- **Never commit the `venv/` folder to Git** - it's already excluded in `.gitignore`
- Always activate the virtual environment before working on the project
- The `requirements.txt` file contains all the project dependencies
- Different developers can use different virtual environment names (venv, env, .venv, etc.)

## Troubleshooting

### Virtual Environment Not Activating

If you can't activate the virtual environment:

- Check that you're in the correct directory
- Verify the virtual environment was created successfully
- On Windows, try running as administrator or check PowerShell execution policy

### Permission Denied Errors

On Windows PowerShell, you might need to change the execution policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Python Command Not Found

- Make sure Python is installed and added to your system PATH
- Try using `python3` instead of `python`
- On Ubuntu/Debian, you might need: `sudo apt install python3-venv`

## Alternative Virtual Environment Tools

While this project uses the built-in `venv` module, you might encounter other tools:

- **virtualenv**: Third-party tool with more features
- **conda**: Popular for data science projects
- **pipenv**: Combines pip and virtualenv with dependency locking
- **poetry**: Modern dependency management with advanced features

## Getting Help

If you encounter issues with the virtual environment setup:

1. Ensure you have the correct Python version installed
2. Check that you're running commands from the project root directory
3. Verify your operating system's specific activation command
4. Consult the [Python venv documentation](https://docs.python.org/3/library/venv.html)

## Quick Start Checklist

- [ ] Clone the repository
- [ ] Navigate to the project directory
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Activate virtual environment: `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows)
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify setup: `python --version` and `pip list`
- [ ] Start developing!

Remember to always activate your virtual environment before working on the project!
