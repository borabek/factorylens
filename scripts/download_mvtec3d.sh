#!/bin/zsh
# Per-category download of MVTec 3D-AD. The single-archive link on the
# official page returns "page not found". On 2026-10-17 the per-category links
# worked for six categories; carrot, cookie, foam and peach 404'd and came from
# the Kaggle mirror gengchensun/mvtec-3d-ad instead (its bagel is
# byte-identical to the official one). Dead links are skipped and listed at
# the end.
set -e
dest=${1:?usage: download_mvtec3d.sh <dest dir>}
mkdir -p "$dest"
cd "$dest"

base=https://www.mydrive.ch/shares
urls=(
  "45891/477cd0a09ea8683b7b77463cc205baa8/download/428793580-1643274607/bagel.tar.xz"
  "45892/8836f2fd7407ec1c54dc165a5b8a2e91/download/428792309-1643274318/cable_gland.tar.xz"
  "45893/4d27bf0566fbd64f05aa34334324b44a/download/428796332-1643275849/carrot.tar.xz"
  "45894/2e94f8ff5f36162f8be266e7b482d1b6/download/428796516-1643276093/cookie.tar.xz"
  "45895/4bc9099f7976819d8d458453e831fc7c/download/428793974-1643275086/dowel.tar.xz"
  "45896/fb990326daa1f3325fa535411bb188c6/download/428797064-1643276186/foam.tar.xz"
  "45897/055d18562435430a64d1edc98da58d2b/download/428797463-1643276448/peach.tar.xz"
  "45898/7b07b3c833a309112d6daca71f31cba2/download/428797964-1643276600/potato.tar.xz"
  "45899/cec7d3a6c2585260ddcc58dd2269e0ee/download/428798351-1643276827/rope.tar.xz"
  "45900/c8261149763d942e033a22a0e8597b5d/download/428798689-1643276973/tire.tar.xz"
)

failed=()
for u in $urls; do
  f=${u:t}
  cat=${f%.tar.xz}
  if [[ -d $cat/test ]]; then
    echo "$cat already extracted"
    continue
  fi
  echo "$(date +%T) downloading $f"
  curl -L -C - --retry 5 --retry-delay 10 -sS -A "Mozilla/5.0" -o $f "$base/$u"
  if [[ $(file -b --mime-type $f) != application/x-xz ]]; then
    echo "$f is not an xz archive (link dead?), skipping" >&2
    rm -f $f
    failed+=($cat)
    continue
  fi
  echo "$(date +%T) extracting $f"
  tar -xJf $f
  rm $f
done
echo "$(date +%T) done, failed: ${failed:-none}"
du -sh .
