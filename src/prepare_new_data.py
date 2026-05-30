"""Clean and prepare new data files for entropy experiment."""
import re
from pathlib import Path

CLEAN_DIR = Path(__file__).resolve().parent.parent / 'data' / 'clean'
RAW_LIT = Path(__file__).resolve().parent.parent / 'data' / 'raw_literature'
RAW_NET = Path(__file__).resolve().parent.parent / 'data' / 'raw_internet_twists.txt'

CLEAN_DIR.mkdir(exist_ok=True)


def clean_literature(text):
    """Strip metadata, headers, footers, website URLs."""
    lines = text.split('\n')
    clean_lines = []
    skip_patterns = [
        r'http[s]?://\S+',
        r'www\.\S+',
        r'电子书', r'书库', r'下载', r'欢迎访问',
        r'本文由.*分享', r'网友分享',
        r'本章完', r'未完待续',
        r'^\d{4}[年.]\d{1,2}[月.]',
        r'早安电子书', r'TXT书库', r'txtsk\.com',
    ]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(re.search(p, line) for p in skip_patterns):
            continue
        if re.match(r'^[\d\s\-_=*]+$', line):
            continue
        clean_lines.append(line)
    return '\n'.join(clean_lines)


# Clean literature files
literature_files = {
    'sanguo': '三国演义.txt',
    'sishitongtang': '四世同堂.txt',
    'tianlongbabu': '天龙八部.txt',
    'bailuyuan': '白鹿原.txt',
}

for key, fname in literature_files.items():
    fpath = RAW_LIT / fname
    if not fpath.exists():
        print(f"SKIP {fname}")
        continue
    text = fpath.read_text(encoding='utf-8')
    cleaned = clean_literature(text)
    out_path = CLEAN_DIR / f'{key}.txt'
    out_path.write_text(cleaned, encoding='utf-8')
    print(f"{fname}: {len(text)} -> {len(cleaned)} chars ({len(cleaned)/len(text)*100:.1f}%)")

# Clean internet twists file
if RAW_NET.exists():
    text = RAW_NET.read_text(encoding='utf-8')
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            continue
        clean_lines.append(line)
    cleaned = '\n'.join(clean_lines)
    out_path = CLEAN_DIR / 'internet_twists.txt'
    out_path.write_text(cleaned, encoding='utf-8')
    print(f"internet_twists: {len(text)} -> {len(cleaned)} chars")

print("\nAll cleaned files:")
for f in sorted(CLEAN_DIR.glob('*.txt')):
    print(f"  {f.name}: {f.stat().st_size} bytes")
