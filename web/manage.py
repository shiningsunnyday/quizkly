#!/usr/bin/env python
import sys

def manage():
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)

if __name__ == "__main__":
    manage()
