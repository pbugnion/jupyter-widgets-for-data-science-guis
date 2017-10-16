
import csv
from collections import namedtuple

INPUT_FILE = '/project/data-initial.csv'

Transaction = namedtuple('Transaction', ['memo', 'category'])

def read_input():
    with open(INPUT_FILE) as f:
        rows = list(csv.DictReader(f))
    transactions = []
    for row in rows:
        transaction = Transaction(
            memo=row['memo'],
            category=row['category']
        )
        transactions.append(transaction)
    return transactions