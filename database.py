import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_NAME = "lulc.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with users and history tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create Users table (Simple mock user for now)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT,
            full_name TEXT,
            email TEXT,
            joined_date TEXT
        )
    ''')

    # Check if password_hash column exists (migration for existing db)
    cursor.execute("PRAGMA table_info(users)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'password_hash' not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    
    # Create History table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            input_image TEXT NOT NULL,
            output_image TEXT NOT NULL,
            stats TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Migration for history table to add stats column
    cursor.execute("PRAGMA table_info(history)")
    hist_columns = [info[1] for info in cursor.fetchall()]
    if 'stats' not in hist_columns:
        cursor.execute("ALTER TABLE history ADD COLUMN stats TEXT")
    
    # Check if default user exists, if not create one
    cursor.execute('SELECT * FROM users WHERE username = ?', ('demo_user',))
    if cursor.fetchone() is None:
        default_pass = generate_password_hash('password123')
        cursor.execute('''
            INSERT INTO users (username, password_hash, full_name, email, joined_date)
            VALUES (?, ?, ?, ?, ?)
        ''', ('demo_user', default_pass, 'LULC Researcher', 'researcher@example.com', datetime.now().strftime("%Y-%m-%d")))
    
    conn.commit()
    conn.close()

def add_prediction(user_id, input_image, output_image, stats=None):
    import json
    conn = get_db_connection()
    cursor = conn.cursor()
    stats_json = json.dumps(stats) if stats else None
    cursor.execute('''
        INSERT INTO history (user_id, input_image, output_image, stats)
        VALUES (?, ?, ?, ?)
    ''', (user_id, input_image, output_image, stats_json))
    conn.commit()
    conn.close()

def get_user_history(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM history WHERE user_id = ? ORDER BY timestamp DESC
    ''', (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    import json
    history_list = []
    for row in rows:
        item = dict(row)
        if item.get('stats'):
            try:
                item['stats'] = json.loads(item['stats'])
            except:
                item['stats'] = None
        history_list.append(item)
    return history_list

def get_history_item(user_id, history_id):
    """
    Fetch one history row for a user (used for the analysis detail page).
    """
    import json

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM history
        WHERE user_id = ? AND id = ?
    ''', (user_id, history_id))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    item = dict(row)
    if item.get('stats'):
        try:
            item['stats'] = json.loads(item['stats'])
        except:
            item['stats'] = None

    return item

def get_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def create_user(username, email, password, full_name=""):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        password_hash = generate_password_hash(password)
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, joined_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, email, password_hash, full_name, datetime.now().strftime("%Y-%m-%d")))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user and user['password_hash']:
        if check_password_hash(user['password_hash'], password):
            return dict(user)
    return None
