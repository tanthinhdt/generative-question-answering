import psycopg2
import json

with open('../../configs/db.json', 'r') as f:
    db_info = json.load(f)

try:
    with psycopg2.connect(**db_info) as conn:
        with