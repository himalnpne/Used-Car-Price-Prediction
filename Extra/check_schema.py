import sqlite3

def check_table_schema():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(history);")
    columns = cursor.fetchall()

    for column in columns:
        print(column)

    conn.close()

if __name__ == '__main__':
    check_table_schema()
