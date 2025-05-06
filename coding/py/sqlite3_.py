import sqlite3 as sl




# automatically creates db if it doesn't exist
con = sl.connect('my-test.db')

with con:
    con.execute("""
        CREATE TABLE USER (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            joiningDate timestamp
        );
    """)

cursor = con.cursor()
query = 'SQL query;'
cursor.execute(query)
result = cursor.fetchall()
print('SQLite Version is {}'.format(result))

# ? is a placeholder - the right way to insert variables into a query. 
# Never use string formatting!!
sql = 'INSERT INTO USER (id, name, age) values(?, ?, ?)'
data = [
    (1, 'Alice', 21),
    (2, 'Bob', 22),
    (3, 'Chris', 23)
]

with con:
    con.executemany(sql, data)

with con:
    data = con.execute("SELECT * FROM USER WHERE age <= 22")
    for row in data:
        print(row)

# Save (commit) the changes
con.commit()

plan = ('water plants', 2)
cursor.execute('SELECT * FROM plan where name=(?)', (plan[0], )) # <--- don't forget the comma
plan_db = cursor.fetchone()
if plan_db:
    plan_id = plan_db[0]

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
# use 'with con:' if you don't want to remember to close the connection
con.close()

#list all tables:
cursor.execute("SELECT name FROM sqlite_master  WHERE type='table';")
print(cursor.fetchall())
# [('done_task',), ('sqlite_sequence',), ('done_date',), ('done_task_date',), ('cycle',)]
'''
SQLite uses an internal table named sqlite_sequence to keep track 
of the largest ROWID. This table is created/initialized only when 
a table containing an AUTOINCREMENT column is created.
'''

# SQLite can seamlessly integrate with Pandas Data Frame.
import pandas as pd
df_skill = pd.DataFrame({
    'user_id': [1,1,2,2,3,3,3],
    'skill': ['Network Security', 'Algorithm Development', 'Network Security', 'Java', 'Python', 'Data Science', 'Machine Learning']
})
df_skill.to_sql('SKILL', con)

df = pd.read_sql('''
    SELECT s.user_id, u.name, u.age, s.skill 
    FROM USER u LEFT JOIN SKILL s ON u.id = s.user_id
''', con)

df.to_sql('USER_SKILL', con)


