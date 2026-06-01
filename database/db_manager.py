import sqlite3
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.security import SecurityManager

DEFAULT_DB_PATH = "face_auth.db"

class DatabaseManager:

    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = db_path
        self.security = SecurityManager()

        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()
        print(f"Database connected successfully at: {self.db_path}")

    def _create_tables(self):

        cursor = self._connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Users (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                username     TEXT UNIQUE NOT NULL,
                display_name TEXT NOT NULL,
                embedding    BLOB NOT NULL,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS AuthLog (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                display_name  TEXT NOT NULL,
                success       INTEGER NOT NULL,
                euclidean     REAL,
                cosine        REAL,
                attempted_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self._connection.commit()

    def save_user(self, display_name, embedding):

        username_hash = self.security.hash_username(display_name.lower().strip())
        embedding_blob = pickle.dumps(embedding)

        try:
            cursor = self._connection.cursor()
            cursor.execute("""
                INSERT INTO Users (username, display_name, embedding)
                VALUES (?, ?, ?)
            """, (username_hash, display_name, embedding_blob))
            self._connection.commit()
            print(f"User profile '{display_name}' saved successfully.")
            return True

        except sqlite3.IntegrityError:
            print(f"User verification error: '{display_name}' already registered.")
            return False

    def get_user(self, display_name):

        username_hash = self.security.hash_username(display_name.lower().strip())

        cursor = self._connection.cursor()
        cursor.execute("""
            SELECT embedding FROM Users WHERE username = ?
        """, (username_hash,))

        row = cursor.fetchone()
        if row is None:
            return None

        return pickle.loads(row[0])

    def user_exists(self, display_name):

        return self.get_user(display_name) is not None

    def get_all_users(self):

        cursor = self._connection.cursor()
        cursor.execute("""
            SELECT id, display_name, created_at FROM Users
            ORDER BY created_at DESC
        """)
        return [
            {'id': r[0], 'display_name': r[1], 'created_at': r[2]}
            for r in cursor.fetchall()
        ]

    def delete_user(self, display_name):

        username_hash = self.security.hash_username(display_name.lower().strip())

        cursor = self._connection.cursor()
        cursor.execute("DELETE FROM Users WHERE username = ?", (username_hash,))
        self._connection.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            print(f"User profile '{display_name}' deleted from database.")
        return deleted

    def log_auth_attempt(self, display_name, success, euclidean=None, cosine=None):

        cursor = self._connection.cursor()
        cursor.execute("""
            INSERT INTO AuthLog (display_name, success, euclidean, cosine)
            VALUES (?, ?, ?, ?)
        """, (display_name, int(success), euclidean, cosine))
        self._connection.commit()

    def get_auth_log(self, display_name=None, limit=50):

        cursor = self._connection.cursor()

        if display_name:
            cursor.execute("""
                SELECT display_name, success, euclidean, cosine, attempted_at
                FROM AuthLog
                WHERE display_name = ?
                ORDER BY attempted_at DESC
                LIMIT ?
            """, (display_name, limit))
        else:
            cursor.execute("""
                SELECT display_name, success, euclidean, cosine, attempted_at
                FROM AuthLog
                ORDER BY attempted_at DESC
                LIMIT ?
            """, (limit,))

        return [
            {
                'display_name': r[0],
                'success'     : bool(r[1]),
                'euclidean'   : r[2],
                'cosine'      : r[3],
                'attempted_at': r[4],
            }
            for r in cursor.fetchall()
        ]

    def get_last_login(self, display_name):

        cursor = self._connection.cursor()
        cursor.execute("""
            SELECT success, attempted_at FROM AuthLog
            WHERE display_name = ? AND success = 1
            ORDER BY attempted_at DESC
            LIMIT 1
        """, (display_name,))

        row = cursor.fetchone()
        if row is None:
            return None

        return {'success': bool(row[0]), 'attempted_at': row[1]}

    def close(self):

        if self._connection:
            self._connection.close()


if __name__ == "__main__":
    import numpy as np

    print("=== Database Subsystem Test Pipeline ===\n")
    db = DatabaseManager("test_face_auth.db")

    # Save mock user structure
    fake_embedding = np.random.rand(20).astype(np.float32)
    db.save_user("Sofia", fake_embedding)

    # Log sequential access scenarios
    db.log_auth_attempt("Sofia", success=True,  euclidean=0.21, cosine=0.95)
    db.log_auth_attempt("Sofia", success=False, euclidean=0.48, cosine=0.81)
    db.log_auth_attempt("Sofia", success=True,  euclidean=0.19, cosine=0.96)

    print("\nSystem Audit Trails:")
    for entry in db.get_auth_log("Sofia"):
        status = "PASS" if entry['success'] else "FAIL"
        print(f"  {status} | EU={entry['euclidean']} | "
              f"Cos={entry['cosine']} | {entry['attempted_at']}")

    print(f"\nLast Successful Session Profile: {db.get_last_login('Sofia')}")

    db.close()
    os.remove("test_face_auth.db")
    print("\nDatabase subsystem validation completed successfully!")