import os
import shutil

def setup_templates():
    # Ensure templates folder exists in the right location
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Copy index.html if it's elsewhere
    if os.path.exists('src/templates/index.html'):
        shutil.copy('src/templates/index.html', 'templates/')

if __name__ == '__main__':
    setup_templates()