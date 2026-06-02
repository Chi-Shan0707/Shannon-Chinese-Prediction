"""
Minimal data cleaning for Chinese entropy experiment.

Rules:
  - Remove ENCODING artifacts only: Wikipedia \1\2 markers, === Sample N ===
  - Remove METADATA headers/footers: website banners, copyright, bylines
  - Remove very short fragments (< 50 CJK chars)
  - KEEP digits, brackets, English-in-Chinese, all natural text
  - Convert GB18030 -> UTF-8 where needed
  - Output to clean/, never modify originals
"""

import re
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
CLEAN_DIR = os.path.join(RAW_DIR, 'clean')

CJK_RANGE = set()
for cp in range(0x4E00, 0x9FFF + 1):
    CJK_RANGE.add(chr(cp))
for cp in range(0x3400, 0x4DBF + 1):
    CJK_RANGE.add(chr(cp))

def is_cjk(c):
    return c in CJK_RANGE

def cjk_count(s):
    return sum(1 for c in s if is_cjk(c))

METADATA_PATTERNS = [
    r'^=== Sample \d+ ===$',
    r'^# .+$',  # comment lines in chinese_samples.txt
    r'^(记者|编辑|校对).*?:',  # reporter credits
    r'^(原标题|特别声明|本文为).*',
    r'^(免责声明|版权).*',
    r'^http[s]?://',
    r'^(扫一扫|扫码下载|下载客户端|关于澎湃).*',
    r'^(阅读原文|查看更多|更多相关内容).*',
    r'^\*版权说明.*',
    r'^(留言|写留言|暂无评论|图片|往期回顾).*',
    r'^WeChat \|',
    r'^文案\|',
    r'^周杰伦.*',
    r'^.*版权说明.*',
    r'^IP SHANGHAI.*',
    r'^\s*$',
    # Standalone short numeric labels (01, 02 in KFC)
    r'^\d{1,3}$',
]

def is_metadata_line(line):
    for pat in METADATA_PATTERNS:
        if re.match(pat, line.strip()):
            return True
    return False

def remove_trailing_bylines(text):
    lines = text.split('\n')
    while lines:
        line = lines[-1].strip()
        if not line:
            lines.pop()
            continue
        stripped = line.strip()
        if len(stripped) < 60 and ('报道' in stripped or '记者' in stripped or '编辑' in stripped or '来源' in stripped):
            lines.pop()
        else:
            break
    return '\n'.join(lines)

def clean_text(text):
    # Remove Wikipedia artifacts
    text = re.sub(r'\\\d', '', text)

    lines = text.split('\n')
    kept_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if is_metadata_line(line):
            continue
        kept_lines.append(stripped)

    text = '\n'.join(kept_lines)
    text = remove_trailing_bylines(text)

    # Split into segments, keep >= 50 CJK chars
    segments = re.split(r'\n\s*\n', text)
    kept_segments = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if cjk_count(seg) >= 50:
            kept_segments.append(seg)

    return '\n\n'.join(kept_segments)


def process_book(filepath, encoding='utf-8'):
    with open(filepath, 'r', encoding=encoding) as f:
        raw = f.read()

    lines = raw.split('\n')
    # Skip TOC/metadata: find first paragraph of ≥50 CJK chars
    content_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if cjk_count(stripped) >= 50 and is_cjk(stripped[0]):
            content_start = max(0, i - 1)
            break

    text = '\n'.join(lines[content_start:])
    return clean_text(text)


def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)

    files = [
        ('human_jianshi.txt', 'gb18030', 'human_jianshi.txt'),
        ('sushi_qiren.txt', 'utf-8', 'sushi_qiren.txt'),
        ('wanli.txt', 'gbk', 'wanli.txt'),
        ('chinese_samples.txt', 'utf-8', 'wiki_samples.txt'),
        ('KFC.txt', 'utf-8', 'kfc_original.txt'),
    ]

    for src_name, encoding, dst_name in files:
        src = os.path.join(RAW_DIR, src_name)
        if not os.path.exists(src):
            print(f"  SKIP {src_name} (not found)")
            continue
        print(f"Processing {src_name} ...")

        if src_name in ('human_jianshi.txt', 'sushi_qiren.txt', 'wanli.txt'):
            cleaned = process_book(src, encoding=encoding)
        else:
            with open(src, 'r', encoding=encoding) as f:
                cleaned = clean_text(f.read())

        dst = os.path.join(CLEAN_DIR, dst_name)
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f"  -> {len(cleaned)} chars, {cjk_count(cleaned)} CJK, saved to {dst}")

    # News files: pass through mostly as-is, just strip metadata
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.startswith('news_') or not fname.endswith('.txt'):
            continue
        src = os.path.join(RAW_DIR, fname)
        print(f"Processing {fname} ...")
        with open(src, 'r', encoding='utf-8') as f:
            cleaned = clean_text(f.read())
        dst = os.path.join(CLEAN_DIR, fname)
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f"  -> {len(cleaned)} chars, {cjk_count(cleaned)} CJK, saved to {dst}")

    # Summary
    print("\n" + "=" * 50)
    print("CLEANING COMPLETE")
    print("=" * 50)
    total_cjk = 0
    for fname in sorted(os.listdir(CLEAN_DIR)):
        with open(os.path.join(CLEAN_DIR, fname), 'r') as f:
            text = f.read()
        cjk = cjk_count(text)
        total_cjk += cjk
        digits = sum(1 for c in text if c.isdigit())
        print(f"  {fname:40s}  {len(text):>8} chars,  {cjk:>7} CJK,  {digits:>4} digits")
    print(f"  {'TOTAL':-<40}>  {total_cjk:>7} CJK")


if __name__ == '__main__':
    main()
