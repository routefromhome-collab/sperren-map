import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote
import pandas as pd
from telegram.ext import Application
from datetime import datetime, timedelta
import re
import os
import sys
import json
import random
from collections import defaultdict
from tqdm.asyncio import tqdm_asyncio
import time

print(os.getcwd())

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
CACHE_FILE = 'cache.json'

MAX_CONCURRENT_REQUESTS = 100
RATE_LIMIT_DELAY = 0.01
CACHE_SAVE_INTERVAL = 50

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
]

def get_random_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

def normalize_street_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r'\bstr\b\.?$', 'straße', name)
    name = re.sub(r'str\.$', 'straße', name)
    name = re.sub(r'str$', 'straße', name)
    name = name.replace('strasse', 'straße')
    return name

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки кэша: {e}. Создаю новый.")
            return {}
    return {}

def save_cache(cache):
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Ошибка сохранения кэша: {e}")

def sort_houses(houses):
    def house_key(h):
        match = re.match(r"(\d+)([a-zA-Z]*)", h)
        if match:
            num, suffix = match.groups()
            return (int(num), suffix)
        return (float('inf'), h)
    return sorted(set(houses), key=house_key)

def group_addresses_by_street(addresses):
    grouped = defaultdict(list)
    for addr in addresses:
        parts = addr.split()
        if len(parts) > 1:
            street = " ".join(parts[:-1])
            house = parts[-1]
            grouped[street].append(house)
        else:
            grouped[addr] = []
    for street in grouped:
        grouped[street] = sort_houses(grouped[street])
    return grouped

def generate_google_maps_urls(streets, city):
    urls = []
    streets = list(streets)
    while len(streets) > 0:
        segment = streets[:20]
        streets = streets[20:]
        encoded_streets = [quote(f"{street}, {city}") for street in segment]
        route_url = '/'.join(encoded_streets)
        urls.append(f"https://www.google.com/maps/dir/{route_url}/")
    return urls

async def safe_request(session, method, url, rate_limiter, **kwargs):
    retries = 5
    for attempt in range(retries):
        try:
            await rate_limiter.acquire()
            
            headers = kwargs.pop('headers', None) or get_random_headers()
            
            async with session.request(method, url, headers=headers, **kwargs) as response:
                if response.status == 429:
                    wait_time = min(2 ** attempt + random.uniform(1, 3), 30)
                    print(f"⚠️ Rate limit (429), ожидание {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    continue
                    
                if response.status >= 500:
                    wait_time = min(2 ** attempt + random.random(), 30)
                    print(f"⚠️ Ошибка сервера {response.status}, повтор через {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    continue
                    
                return await response.text()
                
        except (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError,
                aiohttp.ClientOSError, aiohttp.ClientPayloadError, asyncio.TimeoutError) as e:
            wait = min(2 ** attempt + random.random(), 30)
            if attempt < retries - 1:
                print(f"[safe_request] Ошибка {e.__class__.__name__}, повтор через {wait:.1f}s...")
                await asyncio.sleep(wait)
            else:
                print(f"[safe_request] ❌ Не удалось получить {url} после {retries} попыток")
                return None
        except Exception as e:
            print(f"[safe_request] Непредвиденная ошибка для {url}: {e}")
            return None
    
    return None

class RateLimiter:
    def __init__(self, delay):
        self.delay = delay
        self.last_request = 0
    
    async def acquire(self):
        current = time.time()
        time_since_last = current - self.last_request
        if time_since_last < self.delay:
            await asyncio.sleep(self.delay - time_since_last)
        self.last_request = time.time()

async def fetch_and_parse(session, url, data, street, current_month, current_day, semaphore, cache, rate_limiter, stats):
    async with semaphore:
        cache_key = f"{street}_{current_month}_{current_day}"
        if cache_key in cache:
            stats['cached'] += 1
            return cache[cache_key]

        stats['requests'] += 1
        text = await safe_request(session, "POST", url, rate_limiter, data=data, timeout=aiohttp.ClientTimeout(total=25))
        if not text:
            stats['errors'] += 1
            return None

        soup = BeautifulSoup(text, 'lxml')
        month_divs = soup.find_all('div', class_='month')
        page_text = soup.get_text(" ", strip=True)
        is_calendar = any(div.find('td') for div in month_divs)

        found_addresses = []

        if "Auf dieser Straße gibt es mehrere Abfallkalender" in page_text:
            house_links = soup.select('a[href*="streetname"]')
            for house_link in house_links:
                house_number = house_link.get_text(strip=True)
                if not house_number or house_number in cache.get(street, set()):
                    continue
                house_href = house_link.get('href')
                if not house_href:
                    continue
                full_house_link = urljoin(url, house_href)
                
                stats['requests'] += 1
                house_text = await safe_request(session, "GET", full_house_link, rate_limiter, timeout=aiohttp.ClientTimeout(total=15))
                if not house_text:
                    stats['errors'] += 1
                    continue
                    
                house_soup = BeautifulSoup(house_text, 'lxml')
                for month_div in house_soup.find_all('div', class_='month'):
                    month_header = month_div.find(['h3', 'span'])
                    if not month_header:
                        continue
                    if month_header.get_text(strip=True) != current_month:
                        continue
                    tds = month_div.find_all('td', class_=['', 'holiday', 'exception'])
                    for td in tds:
                        day_text = ''.join(re.findall(r'\d+', td.get_text()))
                        if day_text == current_day and td.find("i", title="Sperrmüll"):
                            found_addresses.append(f"{street} {house_number}")
                            stats['found'] += 1

        if not is_calendar:
            similar_links = soup.select('a[href*="streetname"]')
            exact_match_link = next((l for l in similar_links if l.get_text(strip=True).lower() == normalize_street_name(street)), None)
            if exact_match_link:
                similar_href = exact_match_link.get('href')
                if similar_href:
                    full_link = urljoin(url, similar_href)
                    
                    stats['requests'] += 1
                    sub_text = await safe_request(session, "GET", full_link, rate_limiter, timeout=aiohttp.ClientTimeout(total=15))
                    if sub_text:
                        sub_soup = BeautifulSoup(sub_text, 'lxml')
                        sub_month_divs = sub_soup.find_all('div', class_='month')
                        sub_is_calendar = any(div.find('td') for div in sub_month_divs)
                        if sub_is_calendar:
                            for month_div in sub_month_divs:
                                month_header = month_div.find(['h3', 'span'])
                                if not month_header:
                                    continue
                                if month_header.get_text(strip=True) != current_month:
                                    continue
                                tds = month_div.find_all('td', class_=['', 'holiday', 'exception'])
                                for td in tds:
                                    day_text = ''.join(re.findall(r'\d+', td.get_text()))
                                    if day_text == current_day and td.find("i", title="Sperrmüll"):
                                        found_addresses.append(street)
                                        stats['found'] += 1
                    else:
                        stats['errors'] += 1

        cache[cache_key] = found_addresses if found_addresses else None
        return cache[cache_key]

async def start_parsing(application: Application):
    bot = application.bot
    target_day = "13"
    target_month = "November 2025"
    target_date = datetime.strptime(f"{target_day} {target_month}", "%d %B %Y")
    month_translation = {
        "January": "Januar", "February": "Februar", "March": "März",
        "April": "April", "May": "Mai", "June": "Juni", "July": "Juli",
        "August": "August", "September": "September", "October": "Oktober",
        "November": "November", "December": "Dezember"
    }

    print("📊 Загрузка данных...")
    streets_df = pd.read_excel('2.xlsx', engine='openpyxl')
    streets = streets_df["STRNAME"].str.rstrip('.').to_list()
    print(f"✅ Загружено {len(streets)} улиц")
    
    city = "Wuppertal"
    url = 'https://awg-wuppertal.de/privatkunden/abfallkalender.html'

    cache = load_cache()
    print(f"📦 Загружен кэш: {len(cache)} записей")

    rate_limiter = RateLimiter(RATE_LIMIT_DELAY)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS, ttl_dns_cache=600, force_close=False)
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        text = await safe_request(session, "GET", url, rate_limiter)
        if not text:
            print("❌ Не удалось получить главную страницу")
            return
            
        soup = BeautifulSoup(text, 'lxml')
        form = soup.find('form', attrs={'name': 'demand'})
        if not form:
            await bot.send_message(chat_id=CHAT_ID, text="❌ Форма не найдена.")
            return
        post_url = urljoin(url, form.get('action'))
        inputs = form.find_all('input')
        data = {i.get('name'): i.get('value') or '' for i in inputs}

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        found_whole_streets = set()
        found_houses_overall = defaultdict(set)

        days_ahead = 7
        start_time = time.time()
        
        for day_offset in range(days_ahead):
            current_date = target_date + timedelta(days=day_offset)
            current_day = current_date.strftime("%d").lstrip('0')
            current_month = f"{month_translation[current_date.strftime('%B')]} {current_date.strftime('%Y')}"
            previous_date = current_date - timedelta(days=1)
            previous_day = previous_date.strftime("%d").lstrip('0')
            previous_month = f"{month_translation[previous_date.strftime('%B')]} {previous_date.strftime('%Y')}"

            print(f"\n🔍 Парсинг дня: {previous_day} {previous_month}")
            await bot.send_message(chat_id=CHAT_ID,
                                   text=f"🔍 Начало парсинга данных о сборе крупногабаритного мусора на {previous_day} {previous_month}...")

            streets_to_check = [s for s in streets if s not in found_whole_streets]
            
            stats = {'requests': 0, 'cached': 0, 'errors': 0, 'found': 0}
            tasks = []
            for street in streets_to_check:
                data_copy = data.copy()
                data_copy['tx_bwwastecalendar_pi1[demand][streetname]'] = street
                tasks.append(fetch_and_parse(session, post_url, data_copy, street, current_month, current_day, semaphore, cache, rate_limiter, stats))

            results = await tqdm_asyncio.gather(*tasks, desc=f"🔍 {previous_day} {previous_month}", total=len(tasks))

            if (day_offset + 1) % 2 == 0 or day_offset == days_ahead - 1:
                save_cache(cache)
                print(f"💾 Кэш сохранен ({len(cache)} записей)")

            flattened_results = []
            for res in results:
                if isinstance(res, list):
                    flattened_results.extend(res)
                elif isinstance(res, str):
                    flattened_results.append(res)

            unique_addresses = list(filter(None, flattened_results))

            for item in unique_addresses:
                parts = item.split()
                if len(parts) > 1:
                    house = parts[-1]
                    street_name = " ".join(parts[:-1])
                    found_houses_overall[street_name].add(house)
                else:
                    found_whole_streets.add(item)

            print(f"📊 Статистика: запросов={stats['requests']}, кэш={stats['cached']}, ошибок={stats['errors']}, найдено={stats['found']}")

            if unique_addresses:
                grouped = group_addresses_by_street(unique_addresses)
                message_lines = [f"🗑 Шпера {previous_day} {previous_month}:\n"]
                route_streets = []

                for street_name in sorted(grouped.keys()):
                    houses = grouped[street_name]
                    if houses:
                        for house in houses:
                            link = f"https://www.google.com/maps/search/?api=1&query={quote(f'{street_name} {house}, {city}')}"
                            message_lines.append(f"{street_name} {house} [map]({link})")
                    else:
                        link = f"https://www.google.com/maps/search/?api=1&query={quote(f'{street_name}, {city}')}"
                        message_lines.append(f"{street_name} [map]({link})")
                    route_streets.append(street_name)

                route_urls = generate_google_maps_urls(route_streets, city)
                for i, route_url in enumerate(route_urls, start=1):
                    message_lines.append(f"\n🗺 Маршрут (часть {i}): [map]({route_url})")

                block_size = 50
                for i in range(0, len(message_lines), block_size):
                    await bot.send_message(
                        chat_id=CHAT_ID,
                        text="\n".join(message_lines[i:i + block_size]) + "\n\n#шпера",
                        parse_mode="Markdown",
                        disable_web_page_preview=True
                    )
                    await asyncio.sleep(0.5)
            else:
                await bot.send_message(chat_id=CHAT_ID,
                                       text=f"✅ {previous_day} {previous_month} выходной.\n\n#шпера")

        elapsed = time.time() - start_time
        total_found = sum(len(houses) for houses in found_houses_overall.values()) + len(found_whole_streets)
        
        summary = (
            f"✅ Парсинг завершен!\n"
            f"⏱ Время: {elapsed/60:.1f} мин\n"
            f"🏘 Улиц проверено: {len(streets)}\n"
            f"📍 Найдено адресов: {total_found}\n"
            f"💾 Записей в кэше: {len(cache)}"
        )
        print(f"\n{summary}")
        await bot.send_message(chat_id=CHAT_ID, text=summary)

async def main():
    if not TELEGRAM_BOT_TOKEN or not CHAT_ID:
        print("❌ Ошибка: Не установлены TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID в секретах Replit!")
        print("Добавьте секреты через панель Secrets в Replit.")
        sys.exit(1)
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    await start_parsing(application)

if __name__ == '__main__':
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
