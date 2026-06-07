import aiohttp
import requests
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
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import subprocess
import folium
from geopy.geocoders import Nominatim
import logging
from telegram.error import RetryAfter, TelegramError
import html
import math
from geopy.extra.rate_limiter import RateLimiter as GeoRateLimiter
# ------- Настройки/конфиг -------
# print(os.getcwd())

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # GitHub PAT (optional)
CACHE_FILE = "cache.json"

MAX_CONCURRENT_REQUESTS = 60  # можно поднять/понизить
RATE_LIMIT_DELAY = 0.01
PROGRESS_UPDATE_EVERY = 200  # обновлять прогресс в телеге каждые N улиц
TELEGRAM_CHUNK_MAX = 4000  # безопасный размер сообщения

GEOCODE_TIMEOUT = 10
USER_AGENT = "sperren_route_optimizer"
MAX_POINTS = 300  # безопасный лимит по условию
GOOGLE_CHUNK = 20  # Google Maps waypoints per link
PHOTON_TIMEOUT = 8
OPENCAGE_KEY = " "


# --- Вспомогательные функции ---
def haversine_km(a, b):
    """Возвращает расстояние (км) между парой координат a=(lat,lon), b=(lat,lon)."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    R = 6371.0088
    sa = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(sa))


def build_distance_matrix(coord_list):
    """coord_list: list of (lat,lon). Возвращает матрицу расстояний (km)."""
    n = len(coord_list)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(coord_list[i], coord_list[j])
            mat[i][j] = d
            mat[j][i] = d
    return mat


# --- TSP: nearest neighbor + 2-opt ---
def nearest_neighbor_tsp(dist_mat, start_index=0):
    n = len(dist_mat)
    visited = [False] * n
    route = [start_index]
    visited[start_index] = True
    for _ in range(n - 1):
        last = route[-1]
        # выбрать ближайшую не посещённую
        best = None
        bestd = float("inf")
        for j in range(n):
            if not visited[j] and dist_mat[last][j] < bestd:
                bestd = dist_mat[last][j]
                best = j
        if best is None:
            break
        route.append(best)
        visited[best] = True
    return route


def two_opt(route, dist_mat, improvement_threshold=0.0001):
    """
    Улучшение 2-opt. route — список индексов.
    """
    n = len(route)
    if n < 4:
        return route
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:  # соседние, пропустить
                    continue
                a, b = route[i - 1], route[i]
                c, d = route[j - 1], route[j % n]
                delta = (
                    dist_mat[a][c] + dist_mat[b][d] - dist_mat[a][b] - dist_mat[c][d]
                )
                if delta < -improvement_threshold:
                    # invert
                    route[i:j] = reversed(route[i:j])
                    improved = True
        # можно выйти, если не улучшилось
    return route


USER_AGENTS = [
    # несколько user-agents для ротации
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

# логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------- Утилиты -------
def get_random_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
    }


def normalize_street_name(name: str) -> str:
    import re
    import unicodedata

    s = str(name).strip().lower()

    # Нормализуем unicode (удаляет комбинирующие диакритики, но не меняет ß)
    s = unicodedata.normalize("NFKC", s)

    # Заменяем варианты "str", "str.", "strasse", "strasse." на "straße".
    # Используем word-boundaries, чтобы не менять внутри других слов.
    s = re.sub(r"\bstr(?:\.|asse)?\b", "straße", s)

    # Удаляем/заменяем нежелательную пунктуацию на пробелы.
    # Здесь корректно экранированы символы квадратных скобок и т.д.
    s = re.sub(r'[-/_,\.\\"\'()\[\]]+', " ", s)

    s = re.sub(r"str(?:\.|asse)?(?=[\s,]|$)", "straße", s)

    # Убираем лишние пробелы
    s = re.sub(r"\s+", " ", s).strip()

    return s


def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def chunk_text(text, max_len=TELEGRAM_CHUNK_MAX):
    chunks = []
    while len(text) > max_len:
        # try split at newline
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks


def photon_geocode(query):
    import requests

    try:
        url = "https://photon.komoot.io/api/"
        resp = requests.get(url, params={"q": query, "limit": 1}, timeout=5)
        js = resp.json()
        if js.get("features"):
            geom = js["features"][0]["geometry"]["coordinates"]
            return geom[1], geom[0]
    except:
        return None


def geocode_addresses(addresses, city=None, cache_file="geocode_cache.json"):
    import requests
    import time
    import json
    import random
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderServiceError, GeocoderTimedOut

    # --- Загрузка кэша ---
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except:
        cache = {}

    # ✅ ВАЖНО: RateLimiter на уровне Nominatim
    geolocator = Nominatim(user_agent=USER_AGENT, timeout=15)

    coords = {}

    # ═══════════════════════════════════════════════════════════
    # ФИЛЬТРЫ ДЛЯ ВАЛИДАЦИИ АДРЕСОВ
    # ═══════════════════════════════════════════════════════════
    forbidden_osm_types = {
        "track",
        "path",
        "cycleway",
        "footway",
        "bridleway",
        "construction",
        "transportation",
        "halt",
        "locality",
        "platform",
        "neighbourhood",
    }

    forbidden_keys = [
        "waterway",
        "stream",
        "river",
        "canal",
        "harbour",
        "forest",
        "allotments",
        "leisure",
        "garden",
        "cemetery",
        "industrial",
        "landuse",
        "farm",
        "village_green",
        "nature_reserve",
        "peak",
        "island",
        "islet",
        "public_building",
        "historic",
        "square",
        "locality",
        "platform",
    ]

    def is_valid_address(addr_raw, osm_type=None):
        """Проверяет, является ли адрес валидной улицей/домом"""
        if not isinstance(addr_raw, dict):
            return False

        # ❌ Исключаем недопустимые типы
        if osm_type and osm_type in forbidden_osm_types:
            return False

        # ❌ Исключаем по недопустимым ключам
        if any(addr_raw.get(k) for k in forbidden_keys):
            return False

        # ✅ Принимаем дома с улицами
        if addr_raw.get("house_number") and any(
            addr_raw.get(k) for k in ["road", "street"]
        ):
            return True

        # ✅ Принимаем сами улицы
        if any(addr_raw.get(k) for k in ["road", "street", "residential"]):
            return True

        return False

    def save_cache():
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    def nominatim_geocode(query, attempt=0):
        """Агрессивный backoff при 429 + фильтрация результатов"""
        if attempt > 8:
            logger.error(f"Failed after 8 attempts: {query}")
            return None

        try:
            # ✅ Запрашиваем детали адреса для фильтрации
            locs = geolocator.geocode(query, exactly_one=False, addressdetails=True)

            if not locs:
                return None

            # ✅ ФИЛЬТРУЕМ результаты
            for loc in locs:
                osm_type = loc.raw.get("type")
                address_details = loc.raw.get("address", {})

                if is_valid_address(address_details, osm_type):
                    return (loc.latitude, loc.longitude)

            # Все результаты отфильтрованы
            return None

        except GeocoderTimedOut:
            wait = 10 + attempt * 5
            logger.warning(
                f"Timeout on '{query}' → sleep {wait}s (attempt {attempt + 1})"
            )
            time.sleep(wait)
            return nominatim_geocode(query, attempt + 1)

        except GeocoderServiceError as e:
            if "429" in str(e):
                # ✅ ЖЕСТКИЙ backoff на 429
                wait = 30 + attempt * 15  # 30, 45, 60, 75, 90...
                logger.warning(
                    f"429 Too Many Requests on '{query}' → sleep {wait}s (attempt {attempt + 1})"
                )
                time.sleep(wait)
                return nominatim_geocode(query, attempt + 1)
            else:
                logger.warning(f"Service error: {e}")
                time.sleep(5)
                return None

    def photon_geocode_local(query):
        """Fallback через Photon (менее строгий лимит)"""
        try:
            url = "https://photon.komoot.io/api/"
            resp = requests.get(url, params={"q": query, "limit": 1}, timeout=10)
            js = resp.json()
            if js.get("features"):
                c = js["features"][0]["geometry"]["coordinates"]
                return (c[1], c[0])
        except Exception as e:
            logger.warning(f"Photon error: {e}")
        return None

    total = len(addresses)
    logger.info(f"Geocoding {total} addresses with aggressive backoff and filters...")

    for idx, addr in enumerate(addresses, 1):
        query = f"{addr}, {city}" if city else addr

        # --- КЭШ ПРОВЕРКА ---
        if query in cache:
            coords[addr] = cache[query]
            logger.info(f"[{idx}/{total}] CACHE: {addr}")
            continue

        logger.info(f"[{idx}/{total}] Geocoding: {addr}")

        # --- 1️⃣ Nominatim (с агрессивным backoff и фильтрацией) ---
        result = nominatim_geocode(query)

        # --- 2️⃣ Photon fallback ---
        if not result:
            logger.warning(f"Nominatim failed, trying Photon: {addr}")
            result = photon_geocode_local(query)

        if result:
            coords[addr] = result
            cache[query] = result
            logger.info(f"✅ GEOCODED: {addr} → {result}")
        else:
            coords[addr] = None
            cache[query] = None
            logger.error(f"❌ FAILED: {addr}")

        # --- ОЧЕНЬ БОЛЬШАЯ ПАУЗА между запросами ---
        # Nominatim требует строгого соблюдения: 1 запрос в секунду
        pause = 2.0 + random.uniform(0.5, 1.5)  # 2.5-3.5 сек между запросами
        logger.info(f"Sleeping {pause:.1f}s...")
        time.sleep(pause)

        # --- Сохраняем кэш каждые 10 адресов ---
        if idx % 10 == 0:
            save_cache()
            logger.info(f"✅ Cache saved after {idx} addresses")

    save_cache()
    logger.info(
        f"✅ Geocoding completed: {sum(1 for v in coords.values() if v)} / {total} addresses geocoded"
    )
    return coords


def group_addresses_by_street(addresses):
    grouped = defaultdict(list)
    for addr in addresses:
        # если адрес заканчивается на число или число+букву → есть дом
        m = re.match(r"^(.*?)[\s]+(\d+[a-zA-Z]?)$", addr.strip())
        if m:
            street = m.group(1).strip()
            house = m.group(2).strip()
            grouped[street].append(house)
        else:
            # полностью название улицы
            grouped[addr.strip()] = []

    # сортировка по домам натуральным способом
    def house_key(h):
        m = re.match(r"(\d+)(.*)", h)
        if m:
            return (int(m.group(1)), m.group(2))
        return (10**9, h)

    for s in grouped:
        grouped[s] = sorted(set(grouped[s]), key=house_key)

    return grouped


def generate_google_maps_route_links(ordered_addresses, city, chunk_size=GOOGLE_CHUNK):
    """
    ordered_addresses: list адресов в порядке следования (строки)
    Возвращает список ссылок Google Maps; каждая содержит <= chunk_size точек.
    Формат: https://www.google.com/maps/dir/{A}/{B}/{C}/
    (используем url-encoded адресы)
    """
    links = []
    i = 0
    while i < len(ordered_addresses):
        segment = ordered_addresses[i : i + chunk_size]
        encoded = [quote(f"{s}, {city}") for s in segment]
        route_url = "https://www.google.com/maps/dir/" + "/".join(encoded) + "/"
        links.append(route_url)
        i += chunk_size
    return links


def build_optimal_route(
    addresses_with_houses, start_coords=None, city=None, geocode_pause=15.0
):
    """
    addresses_with_houses: список полных адресов (строк) — те, которые нужно посетить (до 300).
    start_coords:
        - если tuple(lat,lon) — маршрут начнётся ИЗ этой точки
        - если None — маршрут начнётся С ПЕРВОЙ геокодированной точки
    city: город (например "Wuppertal")
    geocode_pause: пауза между геокодами

    ВОЗВРАЩАЕТ: (final_addresses, route_links, coords_map)
    """

    if len(addresses_with_houses) == 0:
        return [], [], {}

    if len(addresses_with_houses) > MAX_POINTS:
        raise ValueError(
            f"Too many points: {len(addresses_with_houses)} > {MAX_POINTS}"
        )

    # 1) Геокодируем все адреса
    coords_map = geocode_addresses(addresses_with_houses, city=city)

    # Фильтруем только те, что нашли координаты
    found_items = [a for a in addresses_with_houses if coords_map.get(a)]
    if not found_items:
        # ничего не геокодируется — вернуть как есть
        return addresses_with_houses, [], coords_map

    coord_list = [coords_map[a] for a in found_items]

    added_start = False
    start_index = 0

    # ✅ Безопасный режим:
    # если start_coords есть — используем как начало
    # если нет — просто начинаем с первой адресной точки, ничего не ломается
    if start_coords:
        coord_list = [start_coords] + coord_list
        found_items = ["__START__"] + found_items
        added_start = True
        start_index = 0
    else:
        start_index = 0  # автоматически: первая точка списка

    # Матрица расстояний
    dist_mat = build_distance_matrix(coord_list)

    # Базовый TSP + улучшение
    initial_route = nearest_neighbor_tsp(dist_mat, start_index=start_index)
    improved = two_opt(initial_route[:], dist_mat)

    # Собираем итоговый список адресов
    ordered = [found_items[i] for i in improved]

    # Если был псевдо-старт __START__, убираем метку
    if added_start:
        if ordered and ordered[0] == "__START__":
            ordered = ordered[1:]

    # Собираем финальный массив
    final_addresses = []
    for a in ordered:
        if a != "__START__":
            final_addresses.append(a)

    # Генерация Google Maps ссылок
    route_links = []
    if start_coords:
        # старт — координаты пользователя
        all_points = [f"{start_coords[0]},{start_coords[1]}"] + [
            f"{a}, {city}" for a in final_addresses
        ]
        i = 0
        while i < len(all_points):
            seg = all_points[i : i + GOOGLE_CHUNK]
            encoded = [quote(s) for s in seg]
            route_links.append(
                "https://www.google.com/maps/dir/" + "/".join(encoded) + "/"
            )
            i += GOOGLE_CHUNK
    else:
        # старт — первая адресная точка
        route_links = generate_google_maps_route_links(
            final_addresses, city, chunk_size=GOOGLE_CHUNK
        )

    return final_addresses, route_links, coords_map


# ------- Rate limiter и безопасный запрос -------
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


async def safe_request(session, method, url, rate_limiter, **kwargs):
    retries = 5
    for attempt in range(retries):
        try:
            await rate_limiter.acquire()
            headers = kwargs.pop("headers", None) or get_random_headers()
            async with session.request(
                method, url, headers=headers, **kwargs
            ) as response:
                if response.status in (429,) or response.status >= 500:
                    wait = min(2**attempt + random.random(), 30)
                    logger.warning(
                        "HTTP %s from %s — wait %.1fs", response.status, url, wait
                    )
                    await asyncio.sleep(wait)
                    continue
                return await response.text()
        except Exception as e:
            wait = min(2**attempt + random.random(), 30)
            logger.warning("Request error %s -> waiting %.1fs", e, wait)
            await asyncio.sleep(wait)
    logger.error("Failed to get %s after %d attempts", url, retries)
    return None


# ------- Telegram helper with RetryAfter handling (HTML parse_mode) -------
async def tg_send(bot, method="send_message", **kwargs):
    """
    Универсальная обёртка: отправляет сообщения, обрабатывает RetryAfter.
    method: 'send_message' or 'send_photo' etc. По умолчанию send_message.
    kwargs — параметры вызова метода.
    """
    max_attempts = 6
    for attempt in range(max_attempts):
        try:
            if method == "send_message":
                return await bot.send_message(**kwargs)
            else:
                # можно расширить при необходимости
                return await getattr(bot, method)(**kwargs)
        except RetryAfter as e:
            wait = e.retry_after + 1
            logger.warning("Telegram RetryAfter: wait %s s", wait)
            await asyncio.sleep(wait)
        except TelegramError as e:
            logger.warning(
                "TelegramError: %s (attempt %d/%d)", e, attempt + 1, max_attempts
            )
            await asyncio.sleep(min(2**attempt, 30))
        except Exception as e:
            logger.exception("Unexpected Telegram exception: %s", e)
            await asyncio.sleep(min(2**attempt, 30))
    logger.error("Failed to send telegram message after retries.")
    return None


# ------- Парсер одной улицы -------
async def fetch_and_parse(
    session,
    url,
    data,
    street,
    current_month,
    current_day,
    semaphore,
    cache,
    rate_limiter,
):
    async with semaphore:
        key = f"{street}__{current_month}__{current_day}"
        if key in cache:
            return cache[key]
        text = await safe_request(
            session,
            "POST",
            url,
            rate_limiter,
            data=data,
            timeout=aiohttp.ClientTimeout(total=25),
        )
        if not text:
            cache[key] = None
            return None

        soup = BeautifulSoup(text, "lxml")
        month_divs = soup.find_all("div", class_="month")
        page_text = soup.get_text(" ", strip=True)
        is_calendar = any(div.find("div") for div in month_divs)
        found_addresses = []

        # страница с выбором домов
        if "Auf dieser Straße gibt es mehrere Abfallkalender" in page_text:
            house_links = soup.select('a[href*="streetname"]')
            for link in house_links:
                house_number = link.get_text(strip=True)
                if not house_number:
                    continue
                full_link = urljoin(url, link.get("href"))
                house_text = await safe_request(
                    session,
                    "GET",
                    full_link,
                    rate_limiter,
                    timeout=aiohttp.ClientTimeout(total=15),
                )
                if not house_text:
                    continue
                house_soup = BeautifulSoup(house_text, "lxml")
                for month_div in house_soup.find_all("div", class_="month"):
                    header = month_div.find(["h3", "span"])
                    if not header:
                        continue
                    if header.get_text(strip=True) != current_month:
                        continue
                    for td in month_div.find_all(
                        "td", class_=["", "holiday", "exception"]
                    ):
                        day_text = "".join(re.findall(r"\d+", td.get_text()))
                        if day_text == current_day and td.find("i", title="Sperrmüll"):
                            # сохраняем полный адрес "Street Name house"
                            found_addresses.append(f"{street} {house_number}")

        # если сразу календарь для всей улицы
        elif is_calendar:
            for month_div in month_divs:
                header = month_div.find(["h3", "span"])

                if not header:
                    continue
                if header.get_text(strip=True) != current_month:
                    continue
                day_divs = month_div.find_all("div", class_="day")

                for day in day_divs:
                    span = day.find("span")
                    if not span:
                        continue
                    day_numbers = re.findall(r"\d+", span.get_text())
                    if not day_numbers:
                        continue
                    day_text = day_numbers[0]
                    if day_text == current_day and day.find("i", title="Sperrmüll"):
                        found_addresses.append(street)
        else:
            # ищем все варианты похожих улиц
            similar_links = soup.select('a[href*="streetname"]')
            for link in similar_links:
                street_text = link.get_text(strip=True)

                # сравнение нормализованное, чтобы исключить ошибки регистра и пробелов
                if normalize_street_name(street_text) == normalize_street_name(street):
                    full_link = urljoin(url, link.get("href"))

                    # скачиваем страницу с реальной улицей
                    exact_page = await safe_request(
                        session,
                        "GET",
                        full_link,
                        rate_limiter,
                        timeout=aiohttp.ClientTimeout(total=15),
                    )

                    if not exact_page:
                        continue

                    exact_soup = BeautifulSoup(exact_page, "lxml")

                    # повторяем стандартный парсинг
                    for month_div in exact_soup.find_all("div", class_="month"):
                        header = month_div.find(["h3", "span"])
                        if not header:
                            continue
                        if header.get_text(strip=True) != current_month:
                            continue
                        day_divs = month_div.find_all("div", class_="day")
                        for day in day_divs:
                            span = day.find("span")
                            if not span:
                                continue
                            day_numbers = re.findall(r"\d+", span.get_text())
                            if not day_numbers:
                                continue
                            day_text = day_numbers[0]
                            if day_text == current_day and day.find(
                                "i", title="Sperrmüll"
                            ):
                                found_addresses.append(street)

        cache[key] = found_addresses if found_addresses else None

        return cache[key]


# ------- Создание карты (включаем ВСЕ адреса) и пуш в Git (аккуратно) -------
def _haversine_m(a, b):
    R = 6371000
    φ1 = math.radians(a[0])
    φ2 = math.radians(b[0])
    dφ = math.radians(b[0] - a[0])
    dλ = math.radians(b[1] - a[1])
    sa = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(sa), math.sqrt(1 - sa))
    return R * c


def create_map_and_push(addresses, city, filename="map.html", geocoded_coords=None):
    import os, json, logging, subprocess, requests, folium, re
    from datetime import datetime
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    from folium.plugins import LocateControl

    logger = logging.getLogger("MAP")
    logger.setLevel(logging.INFO)

    fmap = folium.Map(location=[51.2562, 7.1508], zoom_start=12)

    geocoded_points = []
    geocoded_info = []
    street_names_from_addresses = set()
    street_geometries = []

    # ═══════════════════════════════════════════════════════════
    # 1️⃣ ИСПОЛЬЗУЕМ ПЕРЕДАННЫЕ КООРДИНАТЫ (если есть)
    # ═══════════════════════════════════════════════════════════
    if geocoded_coords:
        for addr in addresses:
            if addr in geocoded_coords and geocoded_coords[addr]:
                lat, lon = geocoded_coords[addr]
                geocoded_points.append([lat, lon])

                google_link = (
                    f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
                )
                popup_html = f"""
                <div style="font-size:14px;">
                    <b>{html.escape(addr)}</b><br><br>
                    <a href="{google_link}" target="_blank"
                       style="
                           display:inline-block;
                           padding:8px 14px;
                           background:#4285F4;
                           color:white;
                           border-radius:6px;
                           text-decoration:none;
                           font-weight:600;
                           box-shadow:0 2px 4px rgba(0,0,0,0.2);
                           transition:0.2s;
                       "
                       onmouseover="this.style.background='#2f6ae1'"
                       onmouseout="this.style.background='#4285F4'"
                    >
                        🔍 Открыть в Google Maps
                    </a>
                </div>
                """
                folium.Marker(
                    [lat, lon], popup=folium.Popup(popup_html, max_width=300)
                ).add_to(fmap)

                geocoded_info.append({"address": addr, "lat": lat, "lon": lon})

                # Извлекаем имя улицы из адреса
                m = re.match(r"^(.*\D)\s+(\d[\dA-Za-z/-]*)\s*$", addr.strip())
                street = normalize_street_name(
                    m.group(1).strip() if m else addr.strip()
                )
                if street:
                    street_names_from_addresses.add(street)

    # ═══════════════════════════════════════════════════════════
    # 2️⃣ FALLBACK: если координат не было (редкий случай)
    # ═══════════════════════════════════════════════════════════
    else:
        geolocator = Nominatim(user_agent="sperren_map_streets")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=5)

        for addr in addresses:
            try:
                q = f"{addr}, {city}" if city else addr

                def is_street_or_house(addr_raw, osm_type=None):
                    if not isinstance(addr_raw, dict):
                        return False
                    forbidden_osm_types = {
                        "track",
                        "path",
                        "cycleway",
                        "footway",
                        "bridleway",
                        "construction",
                        "transportation",
                        "halt",
                        "locality",
                        "platform",
                        "neighbourhood",
                    }
                    if osm_type and osm_type in forbidden_osm_types:
                        return False
                    forbidden_keys = [
                        "waterway",
                        "stream",
                        "river",
                        "canal",
                        "harbour",
                        "forest",
                        "allotments",
                        "leisure",
                        "garden",
                        "cemetery",
                        "industrial",
                        "landuse",
                        "farm",
                        "village_green",
                        "nature_reserve",
                        "peak",
                        "island",
                        "islet",
                        "public_building",
                        "historic",
                        "square",
                        "locality",
                        "platform",
                    ]
                    if any(addr_raw.get(k) for k in forbidden_keys):
                        return False

                    if addr_raw.get("house_number") and any(
                        addr_raw.get(k) for k in ["road", "street"]
                    ):
                        return True
                    if any(addr_raw.get(k) for k in ["road", "street", "residential"]):
                        return True
                    return False

                locs = geolocator.geocode(q, exactly_one=False, addressdetails=True)
                loc = None
                if locs:
                    for l in locs:
                        # print(l.raw)
                        osm_type = l.raw.get("type")
                        if is_street_or_house(l.raw.get("address", {}), osm_type):
                            loc = l
                            break
                if not loc:
                    continue
                lat, lon = float(loc.latitude), float(loc.longitude)
                geocoded_points.append([lat, lon])
                google_link = (
                    f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
                )
                popup_html = f"""
                <div style="font-size:14px;">
                    <b>{html.escape(addr)}</b><br><br>
                    <a href="{google_link}" target="_blank"
                       style="
                           display:inline-block;
                           padding:8px 14px;
                           background:#4285F4;
                           color:white;
                           border-radius:6px;
                           text-decoration:none;
                           font-weight:600;
                           box-shadow:0 2px 4px rgba(0,0,0,0.2);
                           transition:0.2s;
                       "
                       onmouseover="this.style.background='#2f6ae1'"
                       onmouseout="this.style.background='#4285F4'"
                    >
                        🔍 Открыть в Google Maps
                    </a>
                </div>
                """
                folium.Marker(
                    [lat, lon], popup=folium.Popup(popup_html, max_width=300)
                ).add_to(fmap)

                street_from_loc = loc.raw.get("address", {}).get("road")
                geocoded_info.append(
                    {"address": addr, "lat": lat, "lon": lon, "street": street_from_loc}
                )
                if street_from_loc:
                    street_names_from_addresses.add(street_from_loc)
            except:
                continue

    if geocoded_points:
        fmap.fit_bounds(geocoded_points)

    # ═══════════════════════════════════════════════════════════
    # REST остаётся как было (overpass queries и т.д.)
    # ═══════════════════════════════════════════════════════════
    def overpass_query_for_street(name, city_name, timeout=25):
        q = f"""
                    [out:json][timeout:{timeout}];
                    area["name"="{city_name}"]->.a;
                    (
                        way["name"="{name}"]["highway"](area.a);
                        relation["name"="{name}"]["highway"](area.a);
                    );
                    out geom;
                    """
        try:
            r = requests.post(
                "https://overpass-api.de/api/interpreter",
                data=q.encode("utf-8"),
                timeout=timeout,
            )
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return None

    for street in sorted(street_names_from_addresses):
        try:
            data = overpass_query_for_street(street, city)
            coords_all = []
            if data and data.get("elements"):
                for el in data["elements"]:
                    geom = el.get("geometry")
                    if not geom:
                        continue
                    tags = el.get("tags", {})
                    highway_type = el.get("tags", {}).get("highway")
                    if highway_type in [
                        "track",
                        "path",
                        "cycleway",
                        "footway",
                        "bridleway",
                        "construction",
                        "transportation",
                        "halt",
                        "locality",
                        "bus_station",
                        "bus_stop",
                        "station",
                        "platform",
                        "neighbourhood",
                    ]:
                        continue
                    forbidden_keys = [
                        "waterway",
                        "stream",
                        "river",
                        "canal",
                        "harbour",
                        "forest",
                        "allotments",
                        "leisure",
                        "garden",
                        "cemetery",
                        "industrial",
                        "landuse",
                        "farm",
                        "village_green",
                        "nature_reserve",
                        "peak",
                        "island",
                        "islet",
                        "public_building",
                        "historic",
                        "square",
                        "locality",
                        "bus_stop",
                        "bus_station",
                        "station",
                        "platform",
                    ]
                    if any(tags.get(k) for k in forbidden_keys):
                        continue
                    seg = [[float(p["lat"]), float(p["lon"])] for p in geom]
                    if seg:
                        coords_all.extend(seg)
            if coords_all:
                if len(coords_all) > 2000:
                    step = max(1, len(coords_all) // 2000)
                    coords_all = coords_all[::step]
                street_geometries.append(
                    {"name": normalize_street_name(street), "coords": coords_all}
                )
        except:
            continue

    if not street_geometries:
        names_seen = set()
        for gi in geocoded_info:
            nm = normalize_street_name(gi.get("street") or gi.get("address"))
            if not nm or nm in names_seen:
                continue
            names_seen.add(nm)
            street_geometries.append({"name": nm, "coords": [[gi["lat"], gi["lon"]]]})

        if not street_geometries and geocoded_points:
            for i, p in enumerate(geocoded_points):
                street_geometries.append({"name": f"point_{i}", "coords": [p]})

    try:
        LocateControl(auto_start=False).add_to(fmap)
    except:
        pass

    fmap.save(filename)

    # Rest of the function remains the same...
    with open(filename, "r", encoding="utf-8") as fh:
        html_text = fh.read()

    found = re.findall(r"var (\w+) = L\.map", html_text)
    map_var = found[0] if found else None

    streets_json = json.dumps(street_geometries, ensure_ascii=False).replace(
        "</", "<\\/"
    )
    inject_js = f"""
        <script>
        document.addEventListener("DOMContentLoaded", function() {{
            try {{
                var MAP = null;
                {"MAP = " + map_var + ";" if map_var else ""}
                if (!MAP) {{
                    MAP = Object.values(window).find(v => v && v._leaflet_id && v instanceof L.Map);
                }}
                if (!MAP) {{
                    console.error("Map not found");
                    return;
                }}

                var STREETS = {streets_json};

                function haversine_m(a,b) {{
                    var R = 6371000;
                    var φ1 = a[0]*Math.PI/180, φ2 = b[0]*Math.PI/180;
                    var dφ = (b[0]-a[0])*Math.PI/180;
                    var dλ = (b[1]-a[1])*Math.PI/180;
                    var sa = Math.sin(dφ/2)*Math.sin(dφ/2) + Math.cos(φ1)*Math.cos(φ2)*Math.sin(dλ/2)*Math.sin(dλ/2);
                    var c = 2*Math.atan2(Math.sqrt(sa), Math.sqrt(1-sa));
                    return R*c;
                }}

                function closestPointOnSegment(p, a, b) {{
                    var latScale = Math.cos(p[0]*Math.PI/180);
                    var ax=a[0], ay=a[1]*latScale;
                    var bx=b[0], by=b[1]*latScale;
                    var px=p[0], py=p[1]*latScale;
                    var vx=bx-ax, vy=by-ay;
                    var wx=px-ax, wy=py-ay;
                    var vlen2 = vx*vx + vy*vy;
                    if (vlen2===0) return a.slice();
                    var t = (vx*wx + vy*wy) / vlen2;
                    t = Math.max(0, Math.min(1,t));
                    var cx=ax+vx*t, cy=ay+vy*t;
                    return [cx, cy/latScale];
                }}

                function nearestPointOnPolyline(p, coords) {{
                    if (!coords || coords.length===0) return null;
                    var best=null, bestDist=Infinity;
                    if (coords.length===1) {{
                        return {{point:coords[0], dist:haversine_m(p,coords[0])}};
                    }}
                    for (var i=0;i<coords.length-1;i++) {{
                        var a=coords[i], b=coords[i+1];
                        var c=closestPointOnSegment(p,a,b);
                        var d=haversine_m(p,c);
                        if (d<bestDist) {{
                            bestDist=d;
                            best={{point:c,dist:d}};
                        }}
                    }}
                    return best;
                }}

                function showNearestStreet() {{
                    if (!navigator.geolocation) {{
                        alert("Геолокация не поддерживается");
                        return;
                    }}
                    navigator.geolocation.getCurrentPosition(function(pos) {{
                        var user=[pos.coords.latitude,pos.coords.longitude];
                        var bestStreet=null, bestInfo=null;

                        for (var si=0;si<STREETS.length;si++) {{
                            var st=STREETS[si];
                            var coords=st.coords||[];
                            if (!coords.length) continue;                          
                            var flat=(Array.isArray(coords[0]) && Array.isArray(coords[0][0]))? coords.flat():coords;
                            var info=nearestPointOnPolyline(user,flat);
                            if (!info) continue;
                            if (!bestInfo || info.dist<bestInfo.dist) {{
                                bestInfo=info;
                                bestStreet={{name:st.name, coords:flat}};
                            }}
                        }}

                        if (!bestStreet || !bestInfo) {{
                            alert("Не удалось найти ближайшую улицу");
                            return;
                        }}

                        var nearPt=bestInfo.point;
                        var meters=Math.round(bestInfo.dist);
                        var walkMin=Math.max(1, Math.round((meters/1000)/5*60));

                        MAP.flyTo(nearPt,17);

                        if (!window._marks) window._marks=[];
                        window._marks.forEach(function(l){{ try {{ MAP.removeLayer(l); }} catch(e){{}} }});
                        window._marks=[];

                        setTimeout(function(){{
                            var m=L.marker(nearPt).addTo(MAP);
                            m.bindPopup("<div style=\\"font-size:14px;\\">"+
                            "Ближайшая улица: <b>"+(bestStreet.name||"—")+"</b><br><br>"+
                            "Расстояние: "+meters+" м<br>"+
                            "Пешком: примерно "+walkMin+" мин<br><br>"+
                            "<a href=\\"https://www.google.com/maps/search/?api=1&query="+nearPt[0]+","+nearPt[1]+"\\" target=\\"_blank\\""+
                            " style=\\"display:inline-block;padding:8px 14px;background:#4285F4;color:white;border-radius:6px;text-decoration:none;font-weight:600;box-shadow:0 2px 4px rgba(0,0,0,0.2);transition:0.2s;\\""+
                            " onmouseover=\\"this.style.background='#2f6ae1'\\""+
                            " onmouseout=\\"this.style.background='#4285F4'\\""+
                            ">🔍 Открыть в Google Maps</a>"+
                            "</div>").openPopup();
                            window._marks.push(m);
                        }},650);

                        var um=L.circleMarker(user,{{radius:6,color:'#333',fillColor:'#fff',weight:2}}).addTo(MAP);
                        um.bindPopup("Вы здесь");
                        window._marks.push(um);

                    }},function(err){{ alert("Ошибка геопозиции: "+err.message); }},
                    {{enableHighAccuracy:true,timeout:10000,maximumAge:0}});
                }}

                var btn=document.createElement("button");
                btn.textContent="Показать ближайшую улицу ко мне";
                btn.style.position="absolute";
                btn.style.top="10px";
                btn.style.left="50%";
                btn.style.transform="translateX(-50%)";
                btn.style.zIndex=1000;
                btn.style.padding="8px 12px";
                btn.style.background="white";
                btn.style.border="1px solid #888";
                btn.style.borderRadius="6px";
                btn.onclick=showNearestStreet;
                document.body.appendChild(btn);

                var ad=document.createElement("div");
                ad.innerHTML='<div style="font-size:15px;font-weight:bold;">⚡Карта предоставлена телеграм каналом Schwebezeit</div><div style="font-size:13px;">Нажми, перейди и подпишись</div>';
                ad.style.position='fixed';
                ad.style.bottom='20px';
                ad.style.right='20px';
                ad.style.zIndex='9999';
                ad.style.background='white';
                ad.style.borderRadius='12px';
                ad.style.padding='10px 15px';
                ad.style.boxShadow='0 4px 12px rgba(0,0,0,0.15)';
                ad.style.fontFamily='sans-serif';
                ad.style.cursor='pointer';
                ad.style.minWidth='200px';
                ad.onclick=function(){{ window.open('https://t.me/schwebezeit','_blank'); }};
                document.body.appendChild(ad);

            }} catch(e) {{
                console.error("MAP init error:",e);
            }}
        }});
        </script>
        """

    if "</body>" in html_text:
        html_text = html_text.replace("</body>", inject_js + "</body>")
    else:
        html_text += inject_js

    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(html_text)

    # ═══════════════════════════════════════════════════════════
    # Git push (остаётся как было)
    # ══════════���════════════════════════════════════════════════
    if not os.getenv("GITHUB_TOKEN"):
        logger.info("NO GITHUB_TOKEN — saved locally: %s", filename)
        return filename

    TOKEN = os.getenv("GITHUB_TOKEN")
    REPO = "routefromhome-collab/sperren-map"
    BRANCH = "main"
    repo_url = f"https://{TOKEN}@github.com/{REPO}.git"

    try:
        subprocess.run(["git", "config", "--global", "user.email", "bot@example.com"])
        subprocess.run(["git", "config", "--global", "user.name", "MapBot"])

        # обновляем локальную ветку
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", BRANCH])

        # добавляем файл
        subprocess.run(["git", "add", filename])

        # коммитим
        subprocess.run(
            ["git", "commit", "-m", f"Update map {datetime.now().isoformat()}"]
        )

        # пуш с токеном
        if TOKEN:
            repo_url = f"https://{TOKEN}@github.com/{REPO}.git"
            subprocess.run(["git", "push", repo_url, BRANCH])
        else:
            # пуш локальный, если токена нет
            subprocess.run(["git", "push", "origin", BRANCH])

    except Exception as e:
        logger.warning("Git push failed: %s", e)

    url = f"https://routefromhome-collab.github.io/sperren-map/{filename}"
    logger.info("Map available: %s", url)
    return url


# ------- Основной поток парсинга -------
async def start_parsing(application: Application):
    bot = application.bot

    target_date = datetime.strptime("9 June 2026", "%d %B %Y")
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
        "December": "Dezember",
    }

    # read streets
    streets_df = pd.read_excel("2.xlsx", engine="openpyxl")
    streets = streets_df["STRNAME"].str.rstrip(".").to_list()
    # streets = ["Holzer str"]
    city = "Wuppertal"
    url = "https://awg-wuppertal.de/privatkunden/abfallkalender.html"

    cache = load_cache()
    rate_limiter = RateLimiter(RATE_LIMIT_DELAY)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        text = await safe_request(session, "GET", url, rate_limiter)
        if not text:
            await tg_send(
                bot, chat_id=CHAT_ID, text="❌ Не удалось получить страницу с формой."
            )
            return

        soup = BeautifulSoup(text, "lxml")
        form = soup.find("form", attrs={"name": "demand"})
        if not form:
            await tg_send(bot, chat_id=CHAT_ID, text="❌ Форма не найдена.")
            return

        post_url = urljoin(url, form.get("action"))
        data_template = {
            i.get("name"): i.get("value") or "" for i in form.find_all("input")
        }
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        days_ahead = 7
        for day_offset in range(days_ahead):
            current_date = target_date + timedelta(days=day_offset)
            current_day = current_date.strftime("%d").lstrip("0")
            current_month = f"{month_translation[current_date.strftime('%B')]} {current_date.strftime('%Y')}"
            previous_date = current_date - timedelta(days=1)
            previous_day = previous_date.strftime("%d").lstrip("0")
            previous_month = f"{month_translation[previous_date.strftime('%B')]} {previous_date.strftime('%Y')}"

            # уведомление старта
            await tg_send(
                bot,
                chat_id=CHAT_ID,
                text=f"🔍 Начало парсинга {previous_day} {previous_month}...",
            )

            # подготовка задач
            tasks = []
            for street in streets:
                dcopy = data_template.copy()
                dcopy["tx_bwwastecalendar_pi1[demand][streetname]"] = street
                tasks.append(
                    fetch_and_parse(
                        session,
                        post_url,
                        dcopy,
                        street,
                        current_month,
                        current_day,
                        semaphore,
                        cache,
                        rate_limiter,
                    )
                )

            # прогресс: создаём сообщение
            progress_message = None
            try:
                progress_message = await bot.send_message(
                    chat_id=CHAT_ID, text=f"Прогресс: 0/{len(tasks)} улиц..."
                )
            except Exception:
                progress_message = None

            results = []
            idx = 0
            with tqdm(
                total=len(tasks),
                desc=f"Парсинг {previous_day} {previous_month}",
                unit="улиц",
            ) as pbar:
                # асинхронно собираем результаты, обновляем прогресс
                for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
                    res = await coro
                    results.append(res)
                    idx += 1
                    pbar.update(1)
                    # обновление telegram прогресса раз в PROGRESS_UPDATE_EVERY
                    if progress_message and (
                        idx % PROGRESS_UPDATE_EVERY == 0 or idx == len(tasks)
                    ):
                        try:
                            await progress_message.edit_text(
                                f"🔍 Парсинг {previous_day} {previous_month}\nПрогресс: {idx}/{len(tasks)} улиц"
                            )
                        except RetryAfter as e:
                            await asyncio.sleep(e.retry_after + 1)
                        except Exception:
                            pass

            # объединяем результаты
            flattened = []
            for r in results:
                if isinstance(r, list):
                    flattened.extend(r)
                elif isinstance(r, str):
                    flattened.append(r)
            unique_addresses = list(filter(None, flattened))

            if unique_addresses:
                # ═══════════════════════════════════════════════════════════
                # 1️⃣ ИЗВЛЕКАЕМ ТОЛЬКО НАЗВАНИЯ УЛИЦ (для Telegram + маршруты)
                # ═══════════════════════════════════════════════════════════
                streets_only_set = set()
                for addr in unique_addresses:
                    # Парсим "Straße 123" → "Straße"
                    m = re.match(r"^(.*\D)\s+(\d[\dA-Za-z/-]*)\s*$", addr.strip())
                    street_name = m.group(1).strip() if m else addr.strip()
                    streets_only_set.add(street_name)

                # ✅ СОРТИРУЕМ АЛФАВИТНО И СОХРАНЯЕМ
                streets_only_alphabetic = sorted(streets_only_set)

                # ═══════════════════════════════════════════════════════════
                # 2️⃣ ГЕОКОДИРУЕМ И ОПТИМИЗИРУЕМ МАРШРУТ только по улицам
                # ═══════════════════════════════════════════════════════════
                final_streets_optimized, route_urls, coords_map_streets = (
                    build_optimal_route(streets_only_alphabetic, city=city)
                )

                # ═══════════════════════════════════════════════════════════
                # 3️⃣ ФОРМИРУЕМ TELEGRAM СООБЩЕНИЕ: ПО АЛФАВИТУ
                # ═══════════════════════════════════════════════════════════
                lines = [f"🗑 <b>Шпера {previous_day} {previous_month}:</b>\n"]

                # ✅ Выводим В АЛФАВИТНОМ ПОРЯДКЕ (не оптимизированном)
                for street in streets_only_alphabetic:
                    display = html.escape(street)
                    search_url = f"https://www.google.com/maps/search/?api=1&query={quote(f'{street}, {city}')}"
                    lines.append(f'{display} <a href="{search_url}">map</a>')

                # маршруты по ОПТИМИЗИРОВАННЫМ улицам
                for i, route_url in enumerate(route_urls, start=1):
                    lines.append(
                        f'\n🗺 Маршрут (часть {i}): <a href="{route_url}">map</a>'
                    )

                # ═══════════════════════════════════════════════════════════
                # 4️⃣ СОЗДАЁМ КАРТУ со ВСЕМИ адресами (с домами)
                # ═══════════════════════════════════════════════════════════
                filename = f"map_{previous_day}_{previous_month.replace(' ', '_')}.html"

                # Геокодируем ВСЕ адреса (с домами) для карты
                coords_map_all = geocode_addresses(unique_addresses, city=city)

                gh_url = create_map_and_push(
                    unique_addresses, city, filename, geocoded_coords=coords_map_all
                )

                # добавим ссылку на GH Pages
                if gh_url:
                    lines.append(
                        f'\n📍Полная карта на {previous_day} {previous_month}: <a href="{gh_url}">map</a>'
                    )

                # хэштег в конце
                lines.append("\n#шпера")

                full_text = "\n".join(lines)
                # разбиваем на чанки и отсылаем (HTML)
                chunks = chunk_text(full_text, max_len=TELEGRAM_CHUNK_MAX)

                for chunk in chunks:
                    await tg_send(
                        bot,
                        chat_id=CHAT_ID,
                        text=chunk,
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                    await asyncio.sleep(0.3)  # небольшая пауза

            else:
                await tg_send(
                    bot,
                    chat_id=CHAT_ID,
                    text=f"✅ {previous_day} {previous_month} выходной.\n\n#шпера",
                )

            # сохраняем кэш
            save_cache(cache)


# ------- main -------
async def main():
    if not TELEGRAM_BOT_TOKEN or not CHAT_ID:
        logger.error("TELEGRAM_BOT_TOKEN or CHAT_ID not set")
        sys.exit(1)
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    await start_parsing(application)


if __name__ == "__main__":
    # ✅ Проверяем, есть ли тестовый файл
    if os.path.exists("addresses.json"):
        try:
            with open("addresses.json", "r", encoding="utf-8") as f:
                data = json.load(f)

            city = data.get("city", "")
            addresses = data.get("addresses", [])

            if addresses:
                print("📌 Найден addresses.json — создаём карту без запуска парсера...")
                result = create_map_and_push(addresses, city, "debug_map.html")
                print("✅ Карта создана:", result)
                sys.exit()  # ✅ выходим, main() НЕ выполняется
            else:
                print("⚠️ В addresses.json нет поля addresses")

        except Exception as e:
            print("❌ Ошибка чтения addresses.json:", e)

    # ✅ если адресов нет — запускаем обычный парсинг
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
