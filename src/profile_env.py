import argparse
import ctypes
import json
import os
import platform
import re
import struct
import subprocess
import sys
from pathlib import Path
from salt import get_student_params

ROOT = Path(__file__).resolve().parents[1]


def linux_cpuinfo():
    path = Path('/proc/cpuinfo')
    if not path.exists():
        return {}
    data = {}
    for line in path.read_text(errors='replace').splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            data.setdefault(key.strip(), value.strip())
    return data


def windows_cpu():
    if platform.system() != 'Windows':
        return {}
    command = ['powershell', '-NoProfile', '-Command',
               'Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,L2CacheSize,L3CacheSize | ConvertTo-Json -Compress']
    try:
        raw = subprocess.check_output(command, text=True, timeout=20)
        return json.loads(raw)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def linux_caches():
    result = {}
    base = Path('/sys/devices/system/cpu/cpu0/cache')
    for index in sorted(base.glob('index*')):
        def get(filename):
            try:
                return (index / filename).read_text().strip()
            except OSError:
                return None
        level, kind = get('level'), get('type')
        if level and (kind in ('Data', 'Unified')):
            result[f'L{level}_cache'] = get('size')
            if get('coherency_line_size'):
                result['cache_line_bytes'] = int(get('coherency_line_size'))
    return result


def windows_caches():
    """Read data/unified cache descriptors from the Windows topology API."""
    if platform.system() != 'Windows':
        return {}
    from ctypes import wintypes
    api = ctypes.WinDLL('kernel32', use_last_error=True).GetLogicalProcessorInformationEx
    api.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    api.restype = wintypes.BOOL
    length = wintypes.DWORD()
    # RelationCache = 2; the first call obtains the required buffer size.
    if not api(2, None, ctypes.byref(length)) and ctypes.get_last_error() != 122:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(length.value)
    if not api(2, buffer, ctypes.byref(length)):
        raise ctypes.WinError(ctypes.get_last_error())
    offset, sizes, descriptors = 0, set(), []
    while offset < length.value:
        if length.value - offset < 8:
            raise ValueError('Truncated Windows cache record')
        relationship, size = struct.unpack_from('<II', buffer, offset)
        if size < 8 or offset + size > length.value:
            raise ValueError('Invalid Windows cache record size')
        if relationship == 2:
            if size < 20:
                raise ValueError('Truncated Windows cache descriptor')
            level, _, line_size, cache_size, kind = struct.unpack_from('<BBHII', buffer, offset + 8)
            # CacheUnified = 0, CacheData = 2; exclude instruction-only caches.
            if kind in (0, 2):
                descriptors.append({'level': level, 'size_bytes': cache_size,
                                    'line_bytes': line_size})
            if line_size and kind in (0, 2):
                sizes.add(line_size)
        offset += size
    result = {'cache_line_bytes': sizes.pop() if len(sizes) == 1 else None,
              'instances': descriptors,
              'size_scope': 'Per cache instance; heterogeneous sizes are listed separately'}
    for level in (1, 2, 3):
        values = sorted({d['size_bytes'] for d in descriptors if d['level'] == level})
        result[f'L{level}_cache'] = ', '.join(f'{size // 1024}K' for size in values) or None
    return result


def windows_cache_line_bytes():
    return windows_caches().get('cache_line_bytes')


def physical_cores_linux():
    path = Path('/proc/cpuinfo')
    if not path.exists():
        return None
    text = path.read_text(errors='replace')
    identities = set()
    for block in text.split('\n\n'):
        p = re.search(r'^physical id\s*:\s*(\d+)', block, re.M)
        c = re.search(r'^core id\s*:\s*(\d+)', block, re.M)
        if p and c:
            identities.add((p.group(1), c.group(1)))
    return len(identities) or None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student-id', required=True)
    args = parser.parse_args()
    cpu = linux_cpuinfo()
    win = windows_cpu()
    logical = os.cpu_count()
    physical = physical_cores_linux() or win.get('NumberOfCores')
    caches = linux_caches() if platform.system() == 'Linux' else {
        'L1_cache': None, 'L2_cache': f"{win['L2CacheSize']}K" if win.get('L2CacheSize') else None,
        'L3_cache': f"{win['L3CacheSize']}K" if win.get('L3CacheSize') else None,
        'cache_line_bytes': None,
    }
    if platform.system() == 'Windows':
        try:
            caches.update(windows_caches())
        except (OSError, ValueError) as exc:
            print(f'NOTE: Windows cache query failed: {exc}')
    profile = {
        'cpu_model': cpu.get('model name') or win.get('Name') or platform.processor() or 'unknown',
        'microarchitecture': platform.machine(),
        'base_frequency_mhz': float(cpu['cpu MHz']) if 'cpu MHz' in cpu else win.get('MaxClockSpeed'),
        'physical_cores': physical,
        'logical_threads': logical,
        'smt_detected': logical > physical if physical and logical else None,
        'cache': caches,
        'cache_line_bytes': caches.get('cache_line_bytes'),
        'os': platform.system(),
        'kernel_version': platform.release(),
        'platform': platform.platform(),
        'python_implementation': platform.python_implementation(),
        'python_version': sys.version,
    }
    results = ROOT / 'results'
    results.mkdir(exist_ok=True)
    (results / 'host_profile.json').write_text(json.dumps(profile, indent=2), encoding='utf-8')
    params = get_student_params(args.student_id)
    (results / 'salted_params.json').write_text(json.dumps(params, indent=2), encoding='utf-8')
    print(json.dumps({'host_profile': profile, 'salted_params': params}, indent=2))
    if physical is None or any(v is None for v in caches.values()):
        print('NOTE: Some CPU/cache details unavailable from OS interfaces; confirm using a hardware profiler.')


if __name__ == '__main__':
    main()
