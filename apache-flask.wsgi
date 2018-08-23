#! /usr/bin/python
try:
    from app.app import app as application
except ImportError:
    pass
import sys
sys.path.append("/var/www/apache-flask")

