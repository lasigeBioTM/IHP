import atexit
import sys
import sqlite3
import re
__author__ = 'Andre'

conn = sqlite3.connect('data/mirna_tracker.db')
c = conn.cursor()

def load_from_file(path):
    # Create table
    c.execute('''CREATE TABLE mirna
             (acc text, name text, description text)''')
    mirna = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("ID"):
                mirna["name"] = line.split("   ")[1]
            elif line.startswith("AC"):
                mirna["acc"] = line.split("   ")[1][:-1]
            elif line.startswith("DE"):
                mirna["description"] = line.split("   ")[1]
            elif line.startswith("//"):
                t = (mirna["acc"], mirna["name"], mirna["description"])
                c.execute("INSERT INTO mirna VALUES (?,?,?)", t)
                mirna = {}


def find_mirna(term, organism="human"):
    # add hsa if organism is human
    if not term.startswith("hsa-"):
        term = "hsa-" + term
    c.execute("SELECT acc, name FROM mirna WHERE name=?", (term.lower(),))
    return c.fetchone()

def exit_handler():
    conn.commit()
    conn.close()

atexit.register(exit_handler)


def main():
    if sys.argv[1] == "generate": # generate SQLite database for miRNAs
        load_from_file(sys.argv[2]) # path to miRNA tracker file should be the second argument
    else:
        res = find_mirna(sys.argv[1])
        print res

if __name__ == "__main__":
    main()