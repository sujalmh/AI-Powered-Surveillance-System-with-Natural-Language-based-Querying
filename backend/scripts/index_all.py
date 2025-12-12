import json
import sys
import requests

def main():
    # Adjust glob/every_sec as needed
    payload = {
        "glob": "data/clips/camera_0/*.mp4",
        "every_sec": 1.0,
        "with_captions": True,  # enable BLIP-2 captions if available
    }
    url = "http://127.0.0.1:8000/api/semantic/index-all"
    print("POST", url, "payload=", payload)
    try:
        r = requests.post(url, json=payload, timeout=600)
        print("status:", r.status_code)
        try:
            print(json.dumps(r.json(), indent=2))
        except Exception:
            print(r.text)
    except Exception as e:
        print("request error:", repr(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
