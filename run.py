#! /usr/bin/env python
try:
    from app.app import app
except ImportError:
    app = object
    app.run = lambda host: print("error")
    pass
if __name__ == '__main__':
    app.run(host='0.0.0.0')
