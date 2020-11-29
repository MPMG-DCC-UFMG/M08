#!/bin/bash
gunicorn --bind 127.0.0.1:5000 --timeout 300  "M08:create_app()"
