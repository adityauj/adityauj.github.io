#!/usr/bin/env bash
# Site verification gates. Run from the repo root.
set -uo pipefail

fail=0
check() {
  if [ "$2" = "$3" ]; then
    printf '  ok    %-34s %s\n' "$1" "$2"
  else
    printf '  FAIL  %-34s got=%s want=%s\n' "$1" "$2" "$3"
    fail=1
  fi
}

echo "Building..."
hugo --gc --minify >/dev/null || { echo "  FAIL  build"; exit 1; }
echo "Gates:"

check "unrepaired code blocks" \
  "$(grep -ro 'FIXME:code' content/ | wc -l)" 0
check "missing alt text" \
  "$(grep -ro 'FIXME:alt' content/ | wc -l)" 0
check "missing summaries" \
  "$(grep -ro 'FIXME:summary' content/ | wc -l)" 0
check "placeholder filenames" \
  "$(grep -ril 'untitled' content/ public/ | wc -l)" 0
check "post count" \
  "$(find content/posts -name index.md | wc -l)" 5
# --minify collapses the feed onto one line, so count occurrences not lines.
# The home feed also carries About, so scope the check to the posts feed.
check "rss item count" \
  "$(grep -o '<item>' public/posts/index.xml | wc -l)" 5

# Every local <img src> in the built output must resolve to a real file.
img_missing=$(python3 - <<'PY'
import pathlib, re
root = pathlib.Path("public")
missing = []
for page in root.rglob("*.html"):
    html = page.read_text(errors="replace")
    for src in re.findall(r'<img[^>]+src="([^"]+)"', html):
        if src.startswith(("http://", "https://", "data:")):
            continue
        clean = src.split("#")[0].split("?")[0]
        target = root / clean.lstrip("/") if clean.startswith("/") else page.parent / clean
        if not target.is_file():
            missing.append(f"{page}: {src}")
for m in missing:
    print("    " + m)
print(f"COUNT={len(missing)}")
PY
)
echo "$img_missing" | grep -v '^COUNT=' || true
check "unresolved images" "$(echo "$img_missing" | sed -n 's/^COUNT=//p')" 0

# Every alias declared in front matter must have produced a redirect stub.
# Counting all refresh stubs would also catch Hugo's automatic /page/1
# pagination redirects, so check the declared aliases specifically.
alias_missing=$(python3 - <<'PY'
import pathlib, re
missing = []
total = 0
for md in pathlib.Path("content/posts").glob("*/index.md"):
    fm = md.read_text().split("---")[1]
    block = re.search(r"^aliases:\n((?:  - .*\n)+)", fm, re.M)
    if not block:
        continue
    for line in block.group(1).strip().splitlines():
        alias = line.strip()[2:].strip()
        total += 1
        stub = pathlib.Path("public") / alias.strip("/") / "index.html"
        if not stub.is_file():
            missing.append(alias)
for m in missing:
    print("    missing stub: " + m)
print(f"TOTAL={total} MISSING={len(missing)}")
PY
)
echo "$alias_missing" | grep -v '^TOTAL=' || true
check "declared aliases present" \
  "$(echo "$alias_missing" | sed -n 's/.*MISSING=//p')" 0
check "alias count" \
  "$(echo "$alias_missing" | sed -n 's/^TOTAL=\([0-9]*\).*/\1/p')" 10

if [ "$fail" -eq 0 ]; then echo "All gates passed."; else echo "GATES FAILED."; fi
exit "$fail"
