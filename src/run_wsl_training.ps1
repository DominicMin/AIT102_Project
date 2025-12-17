
# Script to run training in WSL (Ubuntu-22.04) for GPU interaction
$WSL_DISTRO = "Ubuntu-22.04"
# Use relative path since we are in src
$BASH_SCRIPT_NAME = "wsl_bootstrap.sh" 

Write-Host "Starting WSL Training Session..." -ForegroundColor Green

# 1. Define the Bash Script Content
# Note: We use the mounted path format. 
$bashContent = @'
#!/bin/bash
# Hardcoded absolute path for safety in this specific environment
PROJECT_ROOT="/mnt/c/Users/DominicMin/Google_Drive/__STUDY__/Courses/3_Python_and_Tensorflow_Programming/AIT102_Project/src"

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "Error: Project directory not found at $PROJECT_ROOT"
    exit 1
fi

cd "$PROJECT_ROOT" || exit
echo "Current Directory: $(pwd)"

# Create venv if not exists
if [ ! -d ".venv_wsl" ]; then
    echo "Creating python virtual environment in .venv_wsl..."
    # Suppress sudo prompt if possible, or hope user has logic. 
    # Actually apt-get might prompt password.
    # To avoid password prompt issues for innocent packages, let's try without sudo first or assume user can type it.
    if ! dpkg -s python3-venv >/dev/null 2>&1; then
        echo "Installing python3-venv (password might be required)..."
        sudo apt-get update && sudo apt-get install -y python3-venv
    fi
    python3 -m venv .venv_wsl
fi

# Activate venv
source .venv_wsl/bin/activate

# Upgrade pip (good practice)
python3 -m pip install --upgrade pip

# Install dependencies (only if TF not found to save time)
if ! python3 -c "import tensorflow" &> /dev/null; then
    echo "Installing TensorFlow with CUDA support..."
    pip install "tensorflow[and-cuda]" pillow tqdm matplotlib
else
    echo "Dependencies already installed."
fi

# Verify GPU
python3 -c "import tensorflow as tf; print('WSL GPU Available:', tf.config.list_physical_devices('GPU'))"

# Run Training
echo "Starting Training..."
python train_model.py
'@

# 2. Write to file with LF line endings (Critical for Bash)
$bashContent = $bashContent -replace "`r`n", "`n"
[System.IO.File]::WriteAllText("$PSScriptRoot\$BASH_SCRIPT_NAME", $bashContent)

# 3. Execute via WSL
Write-Host "Executing $BASH_SCRIPT_NAME inside WSL..."
wsl -d $WSL_DISTRO bash "$BASH_SCRIPT_NAME"

Write-Host "WSL Session Complete." -ForegroundColor Green
