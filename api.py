import requests

API_KEY = "1a05f0acd58a4fd588f8e4500c56d33e"

def get_news():
    url = f"https://newsapi.org/v2/everything?q=news&language=en&apiKey={API_KEY}"
    
    response = requests.get(url)
    data = response.json()

    articles = []

    for article in data['articles']:
        articles.append({
            "title": article['title'],
            "content": article['description'] or article['title']
        })

    return articles