"""
Startup Script for Trading Bot Dashboard
Runs the Streamlit dashboard with proper configuration
"""

import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit', 'yfinance', 'pandas', 'numpy', 
        'sklearn', 'tensorflow', 'plotly', 'ta',
        'websockets', 'statsmodels'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Please install missing packages using: pip install -r requirements.txt")
        return False

    return True

def setup_environment():
    """Setup environment variables and directories"""
    # Create directories if they don't exist
    directories = ['data', 'models', 'dashboard']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Set environment variables if not already set
    if not os.getenv('TRADING_BOT_ENV'):
        os.environ['TRADING_BOT_ENV'] = 'development'

    logger.info("Environment setup complete")

def run_dashboard(port=8501, host='localhost'):
    """Run the Streamlit dashboard"""
    try:
        logger.info(f"Starting Trading Bot Dashboard on {host}:{port}")
        logger.info("Access the dashboard at: http://localhost:8501")

        # Run Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            'dashboard/streamlit_app.py',
            '--server.port', str(port),
            '--server.address', host,
            '--server.headless', 'true'
        ])

    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")

def main():
    """Main function"""
    print("🤖Trading Bot Dashboard Startup")
    print("=" * 40)

    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        sys.exit(1)

    print("✅ Dependencies check passed")

    # Setup environment
    setup_environment()
    print("✅ Environment setup complete")

    # Run dashboard
    print("🚀 Starting dashboard...")
    run_dashboard()

if __name__ == "__main__":
    main()
