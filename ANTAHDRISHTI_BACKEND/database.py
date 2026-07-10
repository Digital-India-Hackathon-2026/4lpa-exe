import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "inspections.db"


def create_database():
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bridge_id TEXT NOT NULL,
            bridge_name TEXT NOT NULL,
            location TEXT NOT NULL,
            image_name TEXT NOT NULL,
            defect_count INTEGER NOT NULL,
            detections TEXT NOT NULL,
            average_confidence REAL NOT NULL,
            priority TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            inspection_time TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_inspection(
    bridge_id,
    bridge_name,
    location,
    image_name,
    defect_count,
    detections,
    average_confidence,
    priority,
    recommendation,
    inspection_time
):
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO inspections (
            bridge_id,
            bridge_name,
            location,
            image_name,
            defect_count,
            detections,
            average_confidence,
            priority,
            recommendation,
            inspection_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        bridge_id,
        bridge_name,
        location,
        image_name,
        defect_count,
        detections,
        average_confidence,
        priority,
        recommendation,
        inspection_time
    ))

    connection.commit()
    connection.close()
def get_inspections():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            bridge_id,
            bridge_name,
            location,
            image_name,
            defect_count,
            detections,
            average_confidence,
            priority,
            recommendation,
            inspection_time
        FROM inspections
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    connection.close()

    return [dict(row) for row in rows]

if __name__ == "__main__":
    create_database()
    print("Database created successfully.")