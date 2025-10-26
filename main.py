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
import time
from tqdm import tqdm
import folium
from geopy.geocoders import Nominatim

print(os.getcwd())

# === Настройки ===
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
CACHE_FILE = 'cache.json'

MAX_CONCURRENT_REQUESTS = 100
RATE_LIMIT_DELAY = 0.01

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
]


# === Вспомогательные функции ===
def get_random_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept":
        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
    }


def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


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
    while streets:
        segment = streets[:20]
        streets = streets[20:]
        encoded_streets = [quote(f"{street}, {city}") for street in segment]
        route_url = '/'.join(encoded_streets)
        urls.append(f"https://www.google.com/maps/dir/{route_url}/")
    return urls


def chunk_text(text, max_len=4000):
    chunks = []
    while len(text) > max_len:
        split_at = text.rfind('\n', 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip('\n')
    if text:
        chunks.append(text)
    return chunks


# === Ограничение запросов ===
class RateLimiter:

    def __init__(self, delay):
        self.delay = delay
        self.last_request = 0

    async def acquire(self):
        current = time.time()
        diff = current - self.last_request
        if diff < self.delay:
            await asyncio.sleep(self.delay - diff)
        self.last_request = time.time()


# === Основной парсинг ===
async def safe_request(session, method, url, rate_limiter, **kwargs):
    retries = 5
    for attempt in range(retries):
        try:
            await rate_limiter.acquire()
            headers = kwargs.pop('headers', None) or get_random_headers()
            async with session.request(method, url, headers=headers,
                                       **kwargs) as response:
                if response.status in [429] or response.status >= 500:
                    await asyncio.sleep(min(2**attempt, 10))
                    continue
                return await response.text()
        except:
            await asyncio.sleep(min(2**attempt, 10))
    return None


async def fetch_and_parse(session, url, data, street, current_month,
                          current_day, semaphore, cache, rate_limiter):
    async with semaphore:
        key = f"{street}_{current_month}_{current_day}"
        if key in cache:
            return cache[key]

        text = await safe_request(session,
                                  "POST",
                                  url,
                                  rate_limiter,
                                  data=data,
                                  timeout=aiohttp.ClientTimeout(total=25))
        if not text:
            cache[key] = None
            return None

        soup = BeautifulSoup(text, 'lxml')
        month_divs = soup.find_all('div', class_='month')
        page_text = soup.get_text(" ", strip=True)
        is_calendar = any(div.find('td') for div in month_divs)
        found_addresses = []

        if "Auf dieser Straße gibt es mehrere Abfallkalender" in page_text:
            house_links = soup.select('a[href*="streetname"]')
            for link in house_links:
                house_number = link.get_text(strip=True)
                if not house_number:
                    continue
                full_link = urljoin(url, link.get('href'))
                house_text = await safe_request(
                    session,
                    "GET",
                    full_link,
                    rate_limiter,
                    timeout=aiohttp.ClientTimeout(total=15))
                if not house_text:
                    continue
                house_soup = BeautifulSoup(house_text, 'lxml')
                for month_div in house_soup.find_all('div', class_='month'):
                    header = month_div.find(['h3', 'span'])
                    if not header: continue
                    if header.get_text(strip=True) != current_month: continue
                    for td in month_div.find_all(
                            'td', class_=['', 'holiday', 'exception']):
                        day_text = ''.join(re.findall(r'\d+', td.get_text()))
                        if day_text == current_day and td.find(
                                "i", title="Sperrmüll"):
                            found_addresses.append(f"{street} {house_number}")
        elif is_calendar:
            for month_div in month_divs:
                header = month_div.find(['h3', 'span'])
                if not header: continue
                if header.get_text(strip=True) != current_month: continue
                for td in month_div.find_all(
                        'td', class_=['', 'holiday', 'exception']):
                    day_text = ''.join(re.findall(r'\d+', td.get_text()))
                    if day_text == current_day and td.find("i",
                                                           title="Sperrmüll"):
                        found_addresses.append(street)

        cache[key] = found_addresses if found_addresses else None
        return cache[key]


# === Генерация карты ===
def create_map(addresses, city, filename="map.html"):
    geolocator = Nominatim(user_agent="ozon_parser_map")
    fmap = folium.Map(location=[51.2562, 7.1508], zoom_start=12)
    for address in addresses:
        try:
            loc = geolocator.geocode(f"{address}, {city}")
            if loc:
                folium.Marker([loc.latitude, loc.longitude],
                              popup=address,
                              icon=folium.Icon(color="blue",
                                               icon="trash",
                                               prefix="fa")).add_to(fmap)
        except Exception:
            continue
    fmap.save(filename)
    return filename


# === Основная логика ===
async def start_parsing(application: Application):
    bot = application.bot
    target_date = datetime.strptime("13 November 2025", "%d %B %Y")

    month_translation = {
        "January": "Januar",
        "February": "Februar",
        "March": "März",
        "April": "April",
        "May": "Mai",
        "June": "Juni",
        "July": "Juli",
        "August": "August",
        "September": "September",
        "October": "Oktober",
        "November": "November",
        "December": "Dezember"
    }

    streets_df = pd.read_excel('2.xlsx', engine='openpyxl')
    streets = streets_df["STRNAME"].str.rstrip('.').to_list()
    city = "Wuppertal"
    url = 'https://awg-wuppertal.de/privatkunden/abfallkalender.html'

    cache = load_cache()
    rate_limiter = RateLimiter(RATE_LIMIT_DELAY)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(connector=connector,
                                     timeout=timeout) as session:
        text = await safe_request(session, "GET", url, rate_limiter)
        soup = BeautifulSoup(text, 'lxml')
        form = soup.find('form', attrs={'name': 'demand'})
        post_url = urljoin(url, form.get('action'))
        data = {
            i.get('name'): i.get('value') or ''
            for i in form.find_all('input')
        }
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        days_ahead = 7

        for day_offset in range(days_ahead):
            current_date = target_date + timedelta(days=day_offset)
            current_day = current_date.strftime("%d").lstrip('0')
            current_month = f"{month_translation[current_date.strftime('%B')]} {current_date.strftime('%Y')}"

            await bot.send_message(
                chat_id=CHAT_ID,
                text=f"🔍 Начало парсинга {current_day} {current_month}...")

            tasks = []
            for street in streets:
                data_copy = data.copy()
                data_copy[
                    'tx_bwwastecalendar_pi1[demand][streetname]'] = street
                tasks.append(
                    fetch_and_parse(session, post_url, data_copy, street,
                                    current_month, current_day, semaphore,
                                    cache, rate_limiter))

            results = []
            with tqdm(total=len(tasks),
                      desc=f"Парсинг {current_day} {current_month}",
                      unit="улиц") as pbar:
                for coro in asyncio.as_completed(tasks):
                    res = await coro
                    results.append(res)
                    pbar.update(1)

            flattened = []
            for r in results:
                if isinstance(r, list): flattened.extend(r)
                elif isinstance(r, str): flattened.append(r)
            unique_addresses = list(filter(None, flattened))

            if unique_addresses:
                grouped = group_addresses_by_street(unique_addresses)
                message_lines = [f"🗑 Шпера {current_day} {current_month}:"]
                route_streets = []

                for street_name in sorted(grouped.keys()):
                    houses = grouped[street_name]
                    if houses:
                        house_links = ", ".join([
                            f"{house} [map](https://www.google.com/maps/search/?api=1&query={quote(f'{street_name} {house}, {city}')})"
                            for house in houses
                        ])
                        message_lines.append(f"{street_name}: {house_links}")
                    else:
                        message_lines.append(
                            f"{street_name} [map](https://www.google.com/maps/search/?api=1&query={quote(f'{street_name}, {city}')})"
                        )
                    route_streets.append(street_name)

                # 💡 Создание карты
                filename = f"map_{current_day}_{current_month.replace(' ', '_')}.html"
                create_map(unique_addresses, city, filename)

                for chunk in chunk_text("\n".join(message_lines)):
                    await bot.send_message(chat_id=CHAT_ID,
                                           text=chunk,
                                           parse_mode="Markdown",
                                           disable_web_page_preview=True)

                await bot.send_message(chat_id=CHAT_ID,
                                       text=f"📍 Карта сохранена: `{filename}`",
                                       parse_mode="Markdown")
            else:
                await bot.send_message(
                    chat_id=CHAT_ID,
                    text=f"✅ {current_day} {current_month} выходной.")

            save_cache(cache)


# === Точка входа ===
async def main():
    if not TELEGRAM_BOT_TOKEN or not CHAT_ID:
        print("❌ TELEGRAM_BOT_TOKEN или CHAT_ID не установлены")
        sys.exit(1)
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    await start_parsing(application)


if __name__ == '__main__':
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
