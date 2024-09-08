
#!/bin/bash

# Update package lists
apt-get update

# Install system dependencies
apt-get install -y python3-pip python3-dev

# Install Python dependencies
pip install -r requirements.txt

# Set up Flask application
export FLASK_APP=backend/app.py

# Run the application
flask run
