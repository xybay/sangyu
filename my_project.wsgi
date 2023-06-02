import sys
import os

sys.path.insert(0, '/home/kangxy/CTAI_flask')
os.environ['FLASK_APP'] = 'app.py'

from app import app as application
