import os
import json
import hashlib

CACHE_FILE = "embed_cache.json"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
else:
    cache = {}


def cache_key(text):
    return hashlib.md5(text.encode()).hexdigest()


def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)