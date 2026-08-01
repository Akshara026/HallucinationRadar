import requests

url = "https://en.wikipedia.org/w/api.php"

params = {
    "action": "query",
    "prop": "extracts",
    "exintro": True,
    "explaintext": True,
    "redirects": 1,
    "titles": "Python (programming language)",
    "format": "json",
}

headers = {
    "User-Agent": "HallucinationRadar/1.0 (your-email@example.com)"
}

r = requests.get(url, params=params, headers=headers)
data = r.json()

page = next(iter(data["query"]["pages"].values()))
print(page["title"])
print(page["extract"][:500])
