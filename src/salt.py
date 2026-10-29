import hashlib
import json
from pathlib import Path


def get_student_params(student_id: str):
    if not student_id.isdigit():
        raise ValueError('Student ID must contain digits only')
    h = int(hashlib.sha256(student_id.encode('utf-8')).hexdigest()[:8], 16)
    return {
        'student_id': student_id,
        'N': 1024 + ((h % 9) * 64),
        'buffer_elements': 16_777_216,
        'seed': h % 100_007,
    }


def read_params():
    path = Path(__file__).resolve().parents[1] / 'results' / 'salted_params.json'
    if not path.exists():
        raise FileNotFoundError('Run src/profile_env.py --student-id YOUR_ID first')
    return json.loads(path.read_text(encoding='utf-8'))
