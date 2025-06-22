import requests

def check_news_on_internet(news_text, api_key):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': news_text,
        'apiKey': api_key,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 5,
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return None

    data = response.json()
    if data.get('totalResults', 0) > 0:
        return data['articles']  # list of articles
    else:
        return []
