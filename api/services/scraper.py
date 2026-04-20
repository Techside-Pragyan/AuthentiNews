import newspaper
from newspaper import Article

def extract_article_details(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp() # Optional: gets summary/keywords
        
        return {
            "title": article.title,
            "text": article.text,
            "authors": article.authors,
            "publish_date": str(article.publish_date),
            "summary": article.summary,
            "keywords": article.keywords,
            "top_image": article.top_image
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None
