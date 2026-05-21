"""
download_negatives.py  —  Download weapon-free images
Uses multiple sources with automatic fallback.
Usage: python download_negatives.py
"""

import os
import urllib.request
import urllib.parse
import json
from pathlib import Path

OUT_DIR = Path("data/negatives/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
#  Source 1 — Wikimedia Commons (free, no login, direct URLs)
# =============================================================================

def download_wikimedia(target=100):
    """
    Search Wikimedia Commons for weapon-free indoor/outdoor scenes.
    Completely free, no API key needed.
    """
    print("\n  [Source 1] Wikimedia Commons")

    searches = [
        "office interior empty",
        "corridor hallway indoor",
        "shopping mall interior",
        "street pedestrian walking",
        "parking lot empty",
        "airport terminal interior",
        "train station interior",
        "lobby interior building",
        "classroom empty",
        "library interior",
        "car interior",
        "traffic street driving",
        "parking lot cars",
    ]

    api_url = "https://commons.wikimedia.org/w/api.php"
    count   = 0

    for query in searches:
        if count >= target:
            break

        params = {
            "action":      "query",
            "generator":   "search",
            "gsrsearch":   f"filetype:bitmap {query}",
            "gsrnamespace": "6",
            "gsrlimit":    "15",
            "prop":        "imageinfo",
            "iiprop":      "url|size",
            "iiurlwidth":  "640",
            "format":      "json",
        }

        url = api_url + "?" + urllib.parse.urlencode(params)
        try:
            req  = urllib.request.Request(
                url, headers={"User-Agent": "WeaponDetection/1.0"}
            )
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())

            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                if count >= target:
                    break
                info = page.get("imageinfo", [{}])[0]
                img_url = info.get("thumburl") or info.get("url", "")
                if not img_url or not any(
                    img_url.lower().endswith(e)
                    for e in [".jpg", ".jpeg", ".png"]
                ):
                    continue

                fname    = f"wikimedia_{count:04d}.jpg"
                out_path = OUT_DIR / fname
                if out_path.exists():
                    count += 1
                    continue

                try:
                    urllib.request.urlretrieve(img_url, str(out_path))
                    count += 1
                    if count % 10 == 0:
                        print(f"    [{count}] downloaded")
                except Exception:
                    continue

        except Exception as e:
            print(f"    [WARN] Query failed: {query} — {e}")
            continue

    print(f"    Total from Wikimedia: {count}")
    return count


# =============================================================================
#  Source 2 — Unsplash (free, no login)
# =============================================================================

def download_unsplash(target=100):
    """
    Download from Unsplash Source API — no key needed.
    """
    print("\n  [Source 2] Unsplash")

    queries = [
        "office", "corridor", "hallway", "lobby",
        "street", "parking", "airport", "classroom",
        "library", "shopping+mall", "train+station",
        "room+interior", "building+interior",
        "car", "traffic", "driving", "parking"
    ]

    count = 0
    for query in queries:
        if count >= target:
            break
        for i in range(8):
            if count >= target:
                break

            # Unsplash source API — returns random image for query
            url      = f"https://source.unsplash.com/640x480/?{query}&sig={count}"
            fname    = f"unsplash_{query}_{i:02d}.jpg"
            out_path = OUT_DIR / fname

            if out_path.exists():
                count += 1
                continue

            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                resp     = urllib.request.urlopen(req, timeout=15)
                img_data = resp.read()

                # Only save if it's actually an image (>10KB)
                if len(img_data) > 10000:
                    with open(out_path, "wb") as f:
                        f.write(img_data)
                    count += 1
                    if count % 10 == 0:
                        print(f"    [{count}] downloaded")

            except Exception:
                continue

    print(f"    Total from Unsplash: {count}")
    return count


# =============================================================================
#  Source 3 — Lorem Picsum (guaranteed working, random scenes)
# =============================================================================

def download_picsum(target=100):
    """
    Lorem Picsum — always works, random high quality photos.
    No weapons, no violence — just random scenic/indoor photos.
    """
    print("\n  [Source 3] Lorem Picsum (guaranteed fallback)")

    count = 0
    # Picsum uses IDs 1-1000 — skip known problematic ones
    skip_ids = {86, 87, 88}   # known NSFW

    for img_id in range(1, 1000):
        if count >= target:
            break
        if img_id in skip_ids:
            continue

        fname    = f"picsum_{img_id:04d}.jpg"
        out_path = OUT_DIR / fname

        if out_path.exists():
            count += 1
            continue

        url = f"https://picsum.photos/id/{img_id}/640/480"
        try:
            req  = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            resp     = urllib.request.urlopen(req, timeout=15)
            img_data = resp.read()

            if len(img_data) > 5000:
                with open(out_path, "wb") as f:
                    f.write(img_data)
                count += 1
                if count % 25 == 0:
                    print(f"    [{count}] downloaded")

        except Exception:
            continue

    print(f"    Total from Picsum: {count}")
    return count


# =============================================================================
#  Main
# =============================================================================

if __name__ == "__main__":
    TARGET = 500

    print(f"\n{'═'*55}")
    print(f"  Downloading {TARGET} weapon-free images")
    print(f"  Output → {OUT_DIR}")
    print(f"{'═'*55}")

    existing = len(list(OUT_DIR.glob("*.jpg")))
    print(f"\n  Already have: {existing} images")

    remaining = max(100, TARGET - existing)
    print(f"  Fetching new car/traffic backgrounds...")

    total = existing

    # Try sources in order — each fills up to TARGET
    per_source = remaining // 3 + 10

    total += download_wikimedia(target=per_source)
    current = len(list(OUT_DIR.glob("*.jpg")))
    print(f"\n  Running total: {current}/{TARGET}")

    total += download_unsplash(target=per_source)
    current = len(list(OUT_DIR.glob("*.jpg")))
    print(f"\n  Running total: {current}")

    total += download_picsum(target=per_source)
    current = len(list(OUT_DIR.glob("*.jpg")))
    print(f"\n  Running total: {current}")

    final = len(list(OUT_DIR.glob("*.jpg")))
    print(f"\n{'═'*55}")
    print(f"  Done!")
    print(f"  Total images : {final}")
    print(f"  Location     : {OUT_DIR}")
    print(f"\n  Next step    : python part4.py")
    print(f"{'═'*55}\n")