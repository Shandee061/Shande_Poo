"""
Módulo para análise de notícias econômicas em tempo real.

Este módulo coleta e analisa notícias de fontes econômicas que podem impactar
o desempenho do WINFUT e outros ativos do mercado brasileiro.
"""
import os
import re
import time
import threading
import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import newspaper
from newspaper import Article
from newspaper import Config
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import trafilatura

from logger import setup_logger

# Setup logger
logger = setup_logger("news_analyzer")

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')


class NewsAnalyzer:
    """
    Classe para análise de notícias econômicas em tempo real que possam impactar
    o desempenho de ativos no mercado brasileiro.
    """
    
    def __init__(self, 
                 news_sources: Optional[List[str]] = None, 
                 keywords: Optional[List[str]] = None,
                 update_interval: int = 600,  # 10 minutos
                 max_articles: int = 50):
        """
        Inicializa o analisador de notícias.
        
        Args:
            news_sources: Lista de URLs de fontes de notícias
            keywords: Lista de palavras-chave para buscar
            update_interval: Intervalo de atualização em segundos
            max_articles: Número máximo de artigos para analisar por vez
        """
        self.logger = logger
        
        # Default sources if none provided (Brazilian financial news sources)
        self.default_sources = [
            "https://www.infomoney.com.br",
            "https://valorinveste.globo.com",
            "https://economia.uol.com.br",
            "https://www.investnews.com.br",
            "https://www.valor.com.br",
            "https://www.moneytimes.com.br",
            "https://www.sunoresearch.com.br"
        ]
        
        # Default keywords if none provided (relevant economic terms in Portuguese)
        self.default_keywords = [
            "ibovespa", "b3", "bolsa", "índice", "winfut", "win", "mini índice",
            "taxa de juros", "selic", "copom", "banco central", "bc", "roberto campos neto",
            "inflação", "ipca", "igpm", "dólar", "câmbio", "pib", "produto interno bruto",
            "fiscal", "orçamento", "governo", "ministério da economia", "tesouro",
            "petrobras", "vale", "itaú", "bradesco", "b3sa"
        ]
        
        # Set sources and keywords
        self.news_sources = news_sources if news_sources else self.default_sources
        self.keywords = keywords if keywords else self.default_keywords
        
        # Setup parameters
        self.update_interval = update_interval
        self.max_articles = max_articles
        
        # Data storage
        self.articles = []
        self.sentiment_trends = {}
        self.impact_scores = {}
        
        # Thread control
        self.running = False
        self.update_thread = None
        
        # Initialize newspaper config
        self.config = Config()
        self.config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        self.config.request_timeout = 10
        self.config.memoize_articles = False
        
        # Initialize sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()
        
        self.logger.info(f"NewsAnalyzer initialized with {len(self.news_sources)} sources and {len(self.keywords)} keywords")
    
    def start(self):
        """Inicia a coleta e análise de notícias em segundo plano."""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            self.logger.info("News analysis started")
    
    def stop(self):
        """Para a coleta e análise de notícias."""
        if self.running:
            self.running = False
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=2.0)
            self.logger.info("News analysis stopped")
    
    def get_latest_news(self, limit: int = 10, keyword_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retorna as notícias mais recentes, opcionalmente filtradas por palavra-chave.
        
        Args:
            limit: Número máximo de notícias para retornar
            keyword_filter: Filtrar notícias que contenham esta palavra-chave
            
        Returns:
            Lista de dicionários com informações das notícias
        """
        if not self.articles:
            return []
        
        filtered_articles = []
        for article in sorted(self.articles, key=lambda x: x.get('date', datetime.datetime(1970, 1, 1)), reverse=True):
            if keyword_filter and not any(keyword_filter.lower() in kw.lower() for kw in article.get('keywords', [])):
                continue
            
            filtered_articles.append({
                'title': article.get('title', 'No title'),
                'source': article.get('source', 'Unknown'),
                'url': article.get('url', ''),
                'date': article.get('date'),
                'summary': article.get('summary', ''),
                'sentiment': article.get('sentiment', {}),
                'keywords': article.get('keywords', [])
            })
            
            if len(filtered_articles) >= limit:
                break
        
        return filtered_articles
    
    def get_impact_scores(self) -> Dict[str, float]:
        """
        Retorna os scores de impacto por palavra-chave.
        
        Returns:
            Dicionário com palavras-chave e seus scores de impacto
        """
        return self.impact_scores
    
    def get_sentiment_trends(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retorna a tendência de sentimento ao longo do tempo.
        
        Returns:
            Dicionário com palavra-chave e lista de valores de sentimento ao longo do tempo
        """
        return self.sentiment_trends
    
    def _update_loop(self):
        """Loop de atualização de notícias em segundo plano."""
        while self.running:
            try:
                self._fetch_and_analyze()
                self._calculate_impact_scores()
                self.logger.info(f"Updated news data. Found {len(self.articles)} articles with relevant keywords.")
            except Exception as e:
                self.logger.error(f"Error in news update loop: {str(e)}")
            
            # Espera pelo próximo ciclo de atualização
            for _ in range(self.update_interval // 5):
                if not self.running:
                    break
                time.sleep(5)
    
    def _fetch_and_analyze(self):
        """Busca e analisa notícias de todas as fontes configuradas."""
        for source_url in self.news_sources:
            try:
                # Parse the source domain
                domain = urlparse(source_url).netloc
                
                # Use trafilatura to get the main content
                try:
                    downloaded = trafilatura.fetch_url(source_url)
                    if not downloaded:
                        self.logger.warning(f"Could not download content from {source_url}")
                        continue
                        
                    source_content = trafilatura.extract(downloaded)
                    
                    # If trafilatura failed, try with requests + BeautifulSoup
                    if not source_content:
                        response = requests.get(source_url, headers={'User-Agent': self.config.browser_user_agent}, timeout=10)
                        if response.status_code != 200:
                            self.logger.warning(f"Failed to access {source_url}, status code: {response.status_code}")
                            continue
                        
                        soup = BeautifulSoup(response.content, 'html.parser')
                        links = soup.find_all('a', href=True)
                    else:
                        # Try to extract links from the trafilatura content
                        soup = BeautifulSoup(downloaded, 'html.parser')
                        links = soup.find_all('a', href=True)
                except Exception as e:
                    self.logger.warning(f"Error fetching source {source_url}: {str(e)}")
                    response = requests.get(source_url, headers={'User-Agent': self.config.browser_user_agent}, timeout=10)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    links = soup.find_all('a', href=True)
                
                # Extract article URLs
                article_urls = []
                for link in links:
                    url = link['href']
                    
                    # Make absolute URL if relative
                    if not url.startswith('http'):
                        if url.startswith('/'):
                            url = f"{source_url.rstrip('/')}{url}"
                        else:
                            url = f"{source_url.rstrip('/')}/{url}"
                    
                    # Basic filtering to avoid non-article pages
                    if '/noticia' in url or '/news' in url or '/article' in url or '/materia' in url:
                        if domain in url and url not in [a.get('url') for a in self.articles]:
                            article_urls.append(url)
                
                # Process a limited number of articles per source
                processed_count = 0
                for url in article_urls[:10]:  # Limit to 10 articles per source
                    try:
                        article_data = self._process_article(url, domain)
                        if article_data and self._contains_keywords(article_data.get('text', '')):
                            # Add the article data to our collection
                            if len(self.articles) >= self.max_articles:
                                # Remove oldest article
                                self.articles = sorted(self.articles, key=lambda x: x.get('date', datetime.datetime(1970, 1, 1)), reverse=True)
                                self.articles.pop()
                            
                            self.articles.append(article_data)
                            self._update_sentiment_trends(article_data)
                            processed_count += 1
                            
                            if processed_count >= 5:  # Limit to 5 relevant articles per source
                                break
                    except Exception as e:
                        self.logger.warning(f"Error processing article {url}: {str(e)}")
                
                self.logger.debug(f"Processed {processed_count} relevant articles from {source_url}")
                
            except Exception as e:
                self.logger.error(f"Error analyzing source {source_url}: {str(e)}")
    
    def _process_article(self, url: str, domain: str) -> Optional[Dict[str, Any]]:
        """
        Processa um artigo para extrair informações relevantes.
        
        Args:
            url: URL do artigo
            domain: Domínio da fonte
            
        Returns:
            Dicionário com dados do artigo ou None se não for possível processar
        """
        try:
            # First try with trafilatura
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                
                if text:
                    # Parse the HTML to get more metadata
                    soup = BeautifulSoup(downloaded, 'html.parser')
                    title = soup.title.string if soup.title else None
                    
                    # Try to extract publication date
                    date = None
                    date_metas = soup.find_all('meta', property=['article:published_time', 'og:published_time'])
                    for meta in date_metas:
                        if 'content' in meta.attrs:
                            try:
                                date = datetime.datetime.fromisoformat(meta['content'].split('+')[0])
                                break
                            except:
                                pass
                    
                    # If date not found in meta, try to extract from the text
                    if not date:
                        date_patterns = [
                            r'(\d{2}/\d{2}/\d{4})',  # 01/01/2023
                            r'(\d{1,2} de [a-zA-Zç]+ de \d{4})'  # 1 de janeiro de 2023
                        ]
                        
                        for pattern in date_patterns:
                            date_match = re.search(pattern, text)
                            if date_match:
                                try:
                                    date_str = date_match.group(1)
                                    if '/' in date_str:
                                        date = datetime.datetime.strptime(date_str, '%d/%m/%Y')
                                    else:
                                        # Convert month name in Portuguese to month number
                                        month_map = {
                                            'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4,
                                            'maio': 5, 'junho': 6, 'julho': 7, 'agosto': 8,
                                            'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12
                                        }
                                        
                                        day, month_name, year = re.match(r'(\d{1,2}) de ([a-zA-Zç]+) de (\d{4})', date_str).groups()
                                        month = month_map.get(month_name.lower())
                                        if month:
                                            date = datetime.datetime(int(year), month, int(day))
                                    break
                                except:
                                    pass
                    
                    # If still no date, use current time
                    if not date:
                        date = datetime.datetime.now()
                    
                    # Get sentiment
                    sentiment = self._analyze_sentiment(text)
                    
                    # Extract keywords
                    keywords = self._extract_keywords(text)
                    
                    # Create a summary (first 3 sentences or 250 chars)
                    sentences = nltk.sent_tokenize(text)
                    summary = ' '.join(sentences[:min(3, len(sentences))])
                    if len(summary) > 250:
                        summary = summary[:247] + '...'
                    
                    return {
                        'title': title,
                        'url': url,
                        'source': domain,
                        'date': date,
                        'text': text,
                        'summary': summary,
                        'sentiment': sentiment,
                        'keywords': keywords
                    }
            
            # If trafilatura fails, try with newspaper
            article = Article(url, language='pt', config=self.config)
            article.download()
            article.parse()
            
            # Only continue if we have some text
            if not article.text:
                return None
            
            # Extract date, defaulting to current date if not available
            date = article.publish_date if article.publish_date else datetime.datetime.now()
            
            # Get sentiment
            sentiment = self._analyze_sentiment(article.text)
            
            # Extract keywords
            keywords = self._extract_keywords(article.text)
            
            # Create a summary (use newspaper's summary if available)
            try:
                article.nlp()
                summary = article.summary
            except:
                sentences = nltk.sent_tokenize(article.text)
                summary = ' '.join(sentences[:min(3, len(sentences))])
            
            if len(summary) > 250:
                summary = summary[:247] + '...'
            
            return {
                'title': article.title,
                'url': url,
                'source': domain,
                'date': date,
                'text': article.text,
                'summary': summary,
                'sentiment': sentiment,
                'keywords': keywords
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to process article {url}: {str(e)}")
            return None
    
    def _contains_keywords(self, text: str) -> bool:
        """Verifica se o texto contém alguma das palavras-chave configuradas."""
        if not text:
            return False
        
        text_lower = text.lower()
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                return True
        
        return False
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extrai palavras-chave do texto que correspondem às configuradas."""
        if not text:
            return []
        
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analisa o sentimento do texto usando VADER (para inglês) e TextBlob (para português).
        
        Args:
            text: Texto a ser analisado
            
        Returns:
            Dicionário com scores de sentimento
        """
        # Combine methods for more reliable sentiment in Portuguese
        try:
            # VADER sentiment (primarily English, but words like "bom"/"ruim" work in many languages)
            vader_scores = self.vader.polarity_scores(text)
            
            # TextBlob sentiment (better for non-English languages)
            tb = TextBlob(text)
            textblob_scores = {'polarity': tb.sentiment.polarity, 'subjectivity': tb.sentiment.subjectivity}
            
            # Combine scores (give more weight to TextBlob for Portuguese)
            combined_compound = (vader_scores['compound'] + 2 * textblob_scores['polarity']) / 3
            
            # Create overall sentiment dictionary
            sentiment = {
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu'],
                'compound': combined_compound,
                'polarity': textblob_scores['polarity'],
                'subjectivity': textblob_scores['subjectivity']
            }
            
            return sentiment
            
        except Exception as e:
            self.logger.warning(f"Error analyzing sentiment: {str(e)}")
            return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1, 'polarity': 0, 'subjectivity': 0}
    
    def _update_sentiment_trends(self, article_data: Dict[str, Any]):
        """
        Atualiza as tendências de sentimento para cada palavra-chave no artigo.
        
        Args:
            article_data: Dados do artigo analisado
        """
        keywords = article_data.get('keywords', [])
        sentiment = article_data.get('sentiment', {})
        date = article_data.get('date', datetime.datetime.now())
        
        for keyword in keywords:
            if keyword not in self.sentiment_trends:
                self.sentiment_trends[keyword] = []
            
            # Add this sentiment data point
            self.sentiment_trends[keyword].append({
                'date': date,
                'sentiment': sentiment.get('compound', 0),
                'article_url': article_data.get('url')
            })
            
            # Keep only the most recent points (maintain a moving window)
            max_points = 100  # Keep no more than 100 data points per keyword
            if len(self.sentiment_trends[keyword]) > max_points:
                # Sort by date and remove oldest
                self.sentiment_trends[keyword] = sorted(
                    self.sentiment_trends[keyword],
                    key=lambda x: x.get('date', datetime.datetime(1970, 1, 1)),
                    reverse=True
                )[:max_points]
    
    def _calculate_impact_scores(self):
        """
        Calcula os scores de impacto para cada palavra-chave com base em:
        - Frequência de menção
        - Sentimento associado
        - Recência da menção
        """
        now = datetime.datetime.now()
        impact_scores = {}
        
        for keyword, trend_data in self.sentiment_trends.items():
            if not trend_data:
                continue
                
            # Calculate frequency score (how often the keyword is mentioned)
            frequency_score = min(1.0, len(trend_data) / 20)  # Cap at 1.0
            
            # Calculate sentiment score (average sentiment weighted by recency)
            total_weighted_sentiment = 0
            total_weight = 0
            
            for data_point in trend_data:
                days_old = (now - data_point.get('date', now)).total_seconds() / 86400
                # More recent articles have higher weight
                recency_weight = 1.0 / (1.0 + days_old)
                sentiment_value = data_point.get('sentiment', 0)
                
                total_weighted_sentiment += sentiment_value * recency_weight
                total_weight += recency_weight
            
            avg_sentiment = total_weighted_sentiment / total_weight if total_weight > 0 else 0
            
            # Calculate recency score (based on most recent mention)
            most_recent = max(trend_data, key=lambda x: x.get('date', datetime.datetime(1970, 1, 1)))
            days_since_recent = (now - most_recent.get('date', now)).total_seconds() / 86400
            recency_score = 1.0 / (1.0 + days_since_recent)
            
            # Combine scores - impact ranges from -1 to 1
            # Frequency affects magnitude, sentiment determines direction, recency amplifies
            impact = avg_sentiment * (0.4 + 0.3 * frequency_score + 0.3 * recency_score)
            
            impact_scores[keyword] = round(impact, 3)
        
        self.impact_scores = impact_scores
    
    def get_news_summary(self) -> Dict[str, Any]:
        """
        Gera um resumo das notícias analisadas e seus impactos.
        
        Returns:
            Dicionário com resumo de notícias e impactos
        """
        if not self.articles:
            return {"status": "No news data available"}
        
        # Get overall market sentiment (average of recent articles)
        recent_articles = sorted(
            self.articles, 
            key=lambda x: x.get('date', datetime.datetime(1970, 1, 1)),
            reverse=True
        )[:min(10, len(self.articles))]
        
        if not recent_articles:
            return {"status": "No recent news available"}
        
        overall_sentiment = sum(a.get('sentiment', {}).get('compound', 0) for a in recent_articles) / len(recent_articles)
        
        # Get most impactful keywords
        top_keywords = sorted(
            [(k, v) for k, v in self.impact_scores.items()],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        # Get most recent articles
        latest_news = self.get_latest_news(limit=3)
        
        return {
            "overall_sentiment": overall_sentiment,
            "sentiment_label": "Positivo" if overall_sentiment > 0.05 else "Negativo" if overall_sentiment < -0.05 else "Neutro",
            "top_keywords": top_keywords,
            "latest_news": latest_news,
            "article_count": len(self.articles),
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


# Função direta para obter texto de uma URL
def get_article_text(url: str) -> Optional[str]:
    """
    Extrai o texto principal de um artigo de notícias a partir da URL.
    
    Args:
        url: URL do artigo
        
    Returns:
        Texto do artigo ou None se falhar
    """
    try:
        # First try with trafilatura (usually better for text extraction)
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                return text
        
        # Fallback to newspaper
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {str(e)}")
        return None


# Test function
if __name__ == "__main__":
    # Example usage
    analyzer = NewsAnalyzer()
    analyzer.start()
    
    # Wait for initial data collection
    time.sleep(10)
    
    # Get latest news
    latest = analyzer.get_latest_news(limit=3)
    
    for article in latest:
        print(f"Title: {article['title']}")
        print(f"Source: {article['source']}")
        print(f"Sentiment: {article['sentiment']['compound']}")
        print(f"Summary: {article['summary']}")
        print("-" * 50)
    
    # Get impact scores
    impacts = analyzer.get_impact_scores()
    for keyword, score in sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"{keyword}: {score}")
    
    analyzer.stop()