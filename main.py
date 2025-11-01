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
print(os.getcwd())

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # GitHub PAT (optional)
CACHE_FILE = 'cache.json'

MAX_CONCURRENT_REQUESTS = 60  # можно поднять/понизить
RATE_LIMIT_DELAY = 0.01
PROGRESS_UPDATE_EVERY = 50  # обновлять прогресс в телеге каждые N улиц
TELEGRAM_CHUNK_MAX = 4000  # безопасный размер сообщения

GEOCODE_TIMEOUT = 10
USER_AGENT = "sperren_route_optimizer"
MAX_POINTS = 300  # безопасный лимит по условию
GOOGLE_CHUNK = 20  # Google Maps waypoints per link


# --- Вспомогательные функции ---
def haversine_km(a, b):
    """Возвращает расстояние (км) между парой координат a=(lat,lon), b=(lat,lon)."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    R = 6371.0088
    sa = math.sin(
        dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return 2 * R * math.asin(math.sqrt(sa))


def geocode_addresses(addresses, city=None, pause=1.0):
    """
        Геокодирует список адресов. Возвращает словарь {address: (lat,lon)}.
        - addresses: список строк (можно до 300)
        - city: добавляется к запросу (например 'Wuppertal')
        - pause: минимальная задержка между запросами к Nominatim
        """
    geolocator = Nominatim(user_agent=USER_AGENT,
                           timeout=GEOCODE_TIMEOUT)  # type: ignore
    geocode = GeoRateLimiter(geolocator.geocode, min_delay_seconds=pause)

    coords = {}
    for addr in addresses:
        query = f"{addr}, {city}" if city else addr
        try:
            loc = geocode(query)
            if loc:
                coords[addr] = (loc.latitude, loc.longitude)
            else:
                coords[addr] = None
        except Exception as e:
            logger.warning("Geocode failed for '%s': %s", addr, e)
            coords[addr] = None

        # небольшая случайная пауза для уважения Nominatim
        time.sleep(random.uniform(0.2, 0.6))

    return coords


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
        bestd = float('inf')
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
                delta = dist_mat[a][c] + dist_mat[b][d] - dist_mat[a][
                    b] - dist_mat[c][d]
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
        "Accept":
        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
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
        except Exception:
            return {}
    return {}


def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def chunk_text(text, max_len=TELEGRAM_CHUNK_MAX):
    chunks = []
    while len(text) > max_len:
        # try split at newline
        split_at = text.rfind('\n', 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip('\n')
    if text:
        chunks.append(text)
    return chunks


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


def generate_google_maps_route_links(ordered_addresses,
                                     city,
                                     chunk_size=GOOGLE_CHUNK):
    """
        ordered_addresses: list адресов в порядке следования (строки)
        Возвращает список ссылок Google Maps; каждая содержит <= chunk_size точек.
        Формат: https://www.google.com/maps/dir/{A}/{B}/{C}/
        (используем url-encoded адресы)
        """
    links = []
    # Google поддерживает origin/destination/waypoints; проще: использовать /dir/A/B/C/...
    i = 0
    while i < len(ordered_addresses):
        segment = ordered_addresses[i:i + chunk_size]
        encoded = [quote(f"{s}, {city}") for s in segment]
        route_url = "https://www.google.com/maps/dir/" + "/".join(
            encoded) + "/"
        links.append(route_url)
        i += chunk_size
    return links


    # --- Высокоуровневая функция: оптимизировать маршрут ---
def build_optimal_route(addresses_with_houses,
                        start_coords=None,
                        city=None,
                        geocode_pause=1.0):
    """
        addresses_with_houses: список полных адресов (строк) — те, которые нужно посетить (до 300).
        start_coords:
            - если tuple(lat,lon) — маршрут начнётся ИЗ этой точки
            - если None — маршрут начнётся С ПЕРВОЙ геокодированной точки
        city: город (например "Wuppertal")
        geocode_pause: пауза между геокодами
        """

    if len(addresses_with_houses) == 0:
        return [], [], {}

    if len(addresses_with_houses) > MAX_POINTS:
        raise ValueError(
            f"Too many points: {len(addresses_with_houses)} > {MAX_POINTS}")

    # 1) Геокодируем все адреса
    coords_map = geocode_addresses(addresses_with_houses,
                                   city=city,
                                   pause=geocode_pause)

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
        all_points = [f"{start_coords[0]},{start_coords[1]}"
                      ] + [f"{a}, {city}" for a in final_addresses]
        i = 0
        while i < len(all_points):
            seg = all_points[i:i + GOOGLE_CHUNK]
            encoded = [quote(s) for s in seg]
            route_links.append("https://www.google.com/maps/dir/" +
                               "/".join(encoded) + "/")
            i += GOOGLE_CHUNK
    else:
        # старт — первая адресная точка
        route_links = generate_google_maps_route_links(final_addresses,
                                                       city,
                                                       chunk_size=GOOGLE_CHUNK)

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
            headers = kwargs.pop('headers', None) or get_random_headers()
            async with session.request(method, url, headers=headers,
                                       **kwargs) as response:
                if response.status in (429, ) or response.status >= 500:
                    wait = min(2**attempt + random.random(), 30)
                    logger.warning("HTTP %s from %s — wait %.1fs",
                                   response.status, url, wait)
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
async def tg_send(bot, method='send_message', **kwargs):
    """
    Универсальная обёртка: отправляет сообщения, обрабатывает RetryAfter.
    method: 'send_message' or 'send_photo' etc. По умолчанию send_message.
    kwargs — параметры вызова метода.
    """
    max_attempts = 6
    for attempt in range(max_attempts):
        try:
            if method == 'send_message':
                return await bot.send_message(**kwargs)
            else:
                # можно расширить при необходимости
                return await getattr(bot, method)(**kwargs)
        except RetryAfter as e:
            wait = e.retry_after + 1
            logger.warning("Telegram RetryAfter: wait %s s", wait)
            await asyncio.sleep(wait)
        except TelegramError as e:
            logger.warning("TelegramError: %s (attempt %d/%d)", e, attempt + 1,
                           max_attempts)
            await asyncio.sleep(min(2**attempt, 30))
        except Exception as e:
            logger.exception("Unexpected Telegram exception: %s", e)
            await asyncio.sleep(min(2**attempt, 30))
    logger.error("Failed to send telegram message after retries.")
    return None


# ------- Парсер одной улицы -------
async def fetch_and_parse(session, url, data, street, current_month,
                          current_day, semaphore, cache, rate_limiter):
    async with semaphore:
        key = f"{street}__{current_month}__{current_day}"
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

        # страница с выбором домов
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
                    if not header:
                        continue
                    if header.get_text(strip=True) != current_month:
                        continue
                    for td in month_div.find_all(
                            'td', class_=['', 'holiday', 'exception']):
                        day_text = ''.join(re.findall(r'\d+', td.get_text()))
                        if day_text == current_day and td.find(
                                "i", title="Sperrmüll"):
                            # сохраняем полный адрес "Street Name house"
                            found_addresses.append(f"{street} {house_number}")

        # если сразу календарь для всей улицы
        elif is_calendar:
            for month_div in month_divs:
                header = month_div.find(['h3', 'span'])
                if not header:
                    continue
                if header.get_text(strip=True) != current_month:
                    continue
                for td in month_div.find_all(
                        'td', class_=['', 'holiday', 'exception']):
                    day_text = ''.join(re.findall(r'\d+', td.get_text()))
                    if day_text == current_day and td.find("i",
                                                           title="Sperrmüll"):
                        # сохраняем целую улицу
                        found_addresses.append(street)

        cache[key] = found_addresses if found_addresses else None
        return cache[key]


# ------- Создание карты (включаем ВСЕ адреса) и пуш в Git (аккуратно) -------


def create_map_and_push(addresses, city, filename="map.html"):
    """
            Создаёт карту:
              - маркеры домов
              - определяет ближайшую улицу ко мне
              - показывает расстояние и примерное время пешком
              - пушит в GH Pages (если GITHUB_TOKEN)
        """
    import os
    import json
    import logging
    import subprocess
    import requests
    import folium
    from datetime import datetime
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    from folium.plugins import LocateControl

    logger = logging.getLogger("MAP")
    logger.setLevel(logging.INFO)

    geolocator = Nominatim(user_agent="sperren_map_streets")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    fmap = folium.Map(location=[51.2562, 7.1508], zoom_start=12)

    geocoded_points = []
    street_names = set()
    street_geometries = []

    import re
    for addr in addresses:
        if isinstance(addr, str):
            m = re.match(r"^(.*\D)\s+(\d[\dA-Za-z/-]*)\s*$", addr.strip())
            street = m.group(1).strip() if m else addr.strip()
            street_names.add(street)

    for addr in addresses:
        try:
            q = f"{addr}, {city}" if city else addr
            loc = geocode(q)
            if not loc:
                continue
            lat, lon = float(loc.latitude), float(loc.longitude)
            geocoded_points.append([lat, lon])
            folium.Marker([lat, lon], popup=addr).add_to(fmap)
        except:
            continue

    if geocoded_points:
        fmap.fit_bounds(geocoded_points)

    # Определяем координаты улиц
    def overpass_query_for_street(name, city, timeout=25):
        q = f"""
                [out:json][timeout:{timeout}];
                area["name"="{city}"]->.a;
                (
                  way["name"="{name}"](area.a);
                  relation["name"="{name}"](area.a);
                );
                out geom;
                """
        try:
            r = requests.post("https://overpass-api.de/api/interpreter",
                              data=q.encode("utf-8"),
                              timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return None

    # Получаем геометрию улиц, но НЕ рисуем их
    for street in sorted(street_names):
        try:
            data = overpass_query_for_street(street, city)
            coords_collected = []

            if data and data.get("elements"):
                for e in data["elements"]:
                    geom = e.get("geometry")
                    if not geom:
                        continue
                    seg = [[p["lat"], p["lon"]] for p in geom]
                    if seg:
                        coords_collected.append(seg)

            if coords_collected:
                mid = coords_collected[0][len(coords_collected[0]) // 2]
                street_geometries.append({
                    "name": street,
                    "coords": coords_collected[0],
                    "mid": mid
                })
                continue

            loc = geolocator.geocode({"street": street, "city": city})
            if loc:
                lat, lon = float(loc.latitude), float(loc.longitude)
                d = 0.00035
                seg = [[lat - d, lon - d], [lat + d, lon + d]]
                street_geometries.append({
                    "name": street,
                    "coords": seg,
                    "mid": [lat, lon]
                })

        except:
            pass

    if geocoded_points and not street_geometries:
        for p in geocoded_points:
            street_geometries.append({
                "name": "point",
                "coords": [p],
                "mid": p
            })

    try:
        LocateControl(auto_start=False).add_to(fmap)
    except:
        pass

    fmap.save(filename)

    with open(filename, "r", encoding="utf-8") as f:
        html = f.read()

    import re
    map_var = re.findall(r"var (\w+) = L\.map", html)
    map_var = map_var[0] if map_var else "map"

    js_streets = []
    for s in street_geometries:
        arr = s["coords"]
        js_streets.append({"name": s["name"], "coords": arr, "mid": s["mid"]})

    # ✅ Добавлен расчёт времени пешком
    inject_js = f"""
        <script>
        document.addEventListener("DOMContentLoaded", function() {{
            window.MAP = {map_var};
            var STREETS = {json.dumps(js_streets)};

            function meterDist(a,b){{
                var R = 6371000;
                var lat1=a[0]*Math.PI/180, lat2=b[0]*Math.PI/180;
                var dLat=(b[0]-a[0])*Math.PI/180, dLon=(b[1]-a[1])*Math.PI/180;
                var s=Math.sin(dLat/2)**2 + Math.cos(lat1)*Math.cos(lat2)*Math.sin(dLon/2)**2;
                return 2*R*Math.atan2(Math.sqrt(s), Math.sqrt(1-s));
            }}

            function showNearestStreet() {{
                if(!navigator.geolocation) return alert("Геолокация не поддерживается");
                navigator.geolocation.getCurrentPosition(function(pos){{
                    var user=[pos.coords.latitude, pos.coords.longitude];
                    var best=null, bestDist=Infinity;

                    STREETS.forEach(s => {{
                        if(!s.mid) return;
                        var d=meterDist(user, s.mid);
                        if(d<bestDist) {{ bestDist=d; best=s; }}
                    }});

                    if(!best) return alert("Не удалось найти ближайшую улицу");

                    // ✅ Время пешком
                    var walkTimeSec = bestDist / 1.4;
                    var walkMin = Math.max(1, Math.round(walkTimeSec / 60));

                    L.marker(best.mid).addTo(MAP).bindPopup(
                        "Улица: <b>"+best.name+
                        "</b><br>Расстояние: "+Math.round(bestDist)+" м"+
                        "<br>Пешком примерно: "+walkMin+" мин."
                    ).openPopup();

                    L.circleMarker(user, {{radius:6,color:'#000',fillColor:'#fff',weight:2}}).addTo(MAP).bindPopup("Вы здесь");
                    MAP.flyTo(best.mid, 16);
                }}, err => alert("Ошибка геопозиции: "+err.message), {{enableHighAccuracy:true}});
            }}

            var b=document.createElement("button");
            b.textContent="Показать ближайшую улицу ко мне";
            b.style.position="absolute"; b.style.top="10px"; b.style.left="50%";
            b.style.transform="translateX(-50%)";
            b.style.zIndex=999999; b.style.padding="8px 12px";
            b.style.background="white"; b.style.border="1px solid #666";
            b.style.borderRadius="6px";
            b.onclick=showNearestStreet;
            document.body.appendChild(b);
        }});
        </script>
        </body>
        """

    html = html.replace("</body>", inject_js)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    # GitHub push
    if not os.getenv("GITHUB_TOKEN"):
        print("Карта сохранена локально:", filename)
        return filename

    TOKEN = os.getenv("GITHUB_TOKEN")
    REPO = "routefromhome-collab/sperren-map"
    BRANCH = "main"
    repo_url = f"https://{TOKEN}@github.com/{REPO}.git"

    try:
        subprocess.run(
            ["git", "config", "--global", "user.email", "bot@example.com"])
        subprocess.run(["git", "config", "--global", "user.name", "MapBot"])
        subprocess.run(
            ["git", "pull", "--rebase", "--autostash", repo_url, BRANCH])
        subprocess.run(["git", "add", filename])
        subprocess.run([
            "git", "commit", "-m", f"Update map {datetime.now().isoformat()}"
        ])
        subprocess.run(["git", "push", repo_url, BRANCH])
    except Exception as e:
        print("Git push failed:", e)

    url = f"https://routefromhome-collab.github.io/sperren-map/{filename}"
    print("Map available:", url)
    return url


# ------- Основной поток парсинга -------
async def start_parsing(application: Application):
    bot = application.bot

    target_date = datetime.strptime("18 November 2025", "%d %B %Y")
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

    # read streets
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
        if not text:
            await tg_send(bot,
                          chat_id=CHAT_ID,
                          text="❌ Не удалось получить страницу с формой.")
            return

        soup = BeautifulSoup(text, 'lxml')
        form = soup.find('form', attrs={'name': 'demand'})
        if not form:
            await tg_send(bot, chat_id=CHAT_ID, text="❌ Форма не найдена.")
            return

        post_url = urljoin(url, form.get('action'))
        data_template = {
            i.get('name'): i.get('value') or ''
            for i in form.find_all('input')
        }
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        days_ahead = 7
        for day_offset in range(days_ahead):
            current_date = target_date + timedelta(days=day_offset)
            current_day = current_date.strftime("%d").lstrip('0')
            current_month = f"{month_translation[current_date.strftime('%B')]} {current_date.strftime('%Y')}"
            previous_date = current_date - timedelta(days=1)
            previous_day = previous_date.strftime("%d").lstrip('0')
            previous_month = f"{month_translation[previous_date.strftime('%B')]} {previous_date.strftime('%Y')}"

            # уведомление старта
            await tg_send(
                bot,
                chat_id=CHAT_ID,
                text=f"🔍 Начало парсинга {previous_day} {previous_month}...")

            # подготовка задач
            tasks = []
            for street in streets:
                dcopy = data_template.copy()
                dcopy['tx_bwwastecalendar_pi1[demand][streetname]'] = street
                tasks.append(
                    fetch_and_parse(session, post_url, dcopy, street,
                                    current_month, current_day, semaphore,
                                    cache, rate_limiter))

            # прогресс: создаём сообщение
            progress_message = None
            try:
                progress_message = await bot.send_message(
                    chat_id=CHAT_ID, text=f"Прогресс: 0/{len(tasks)} улиц...")
            except Exception:
                progress_message = None

            results = []
            idx = 0
            with tqdm(total=len(tasks),
                      desc=f"Парсинг {previous_day} {previous_month}",
                      unit="улиц") as pbar:
                # асинхронно собираем результаты, обновляем прогресс
                for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
                    res = await coro
                    results.append(res)
                    idx += 1
                    pbar.update(1)
                    # обновление telegram прогресса раз в PROGRESS_UPDATE_EVERY
                    if progress_message and (idx % PROGRESS_UPDATE_EVERY == 0
                                             or idx == len(tasks)):
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

            # группируем по улицам (ключи — полные названия улиц)
            grouped = group_addresses_by_street(unique_addresses)
            streets_only = sorted(grouped.keys())

            if streets_only:
                # Формируем единый текст (HTML)
                lines = [f"🗑 <b>Шпера {previous_day} {previous_month}:</b>\n"]
                for street_name in streets_only:
                    # отображаем полное имя (экранируем HTML)
                    display = html.escape(street_name)
                    search_url = f"https://www.google.com/maps/search/?api=1&query={quote(f'{street_name}, {city}')}"
                    # HTML link: <a href="url">text</a>
                    lines.append(f'{display} <a href="{search_url}">map</a>')

                # маршруты по улицам (разбиваем на части по 20 улиц)
                ordered, route_urls, coords_map = build_optimal_route(
                    streets_only, city=city)

                for i, route_url in enumerate(route_urls, start=1):
                    # вставляем в одну строку с пометкой "map (URL)"
                    # в HTML: делаем кликабельную ссылку
                    lines.append(
                        f'\n🗺 Маршрут (часть {i}): <a href="{route_url}">map</a>'
                    )

                # создаём карту со всеми адресами (включая дома) и пушим
                filename = f"map_{previous_day}_{previous_month.replace(' ', '_')}.html"
                gh_url = create_map_and_push(unique_addresses, city, filename)

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
                    await tg_send(bot,
                                  chat_id=CHAT_ID,
                                  text=chunk,
                                  parse_mode="HTML",
                                  disable_web_page_preview=True)
                    await asyncio.sleep(0.3)  # небольшая пауза

            else:
                await tg_send(
                    bot,
                    chat_id=CHAT_ID,
                    text=
                    f"✅ {previous_day} {previous_month} выходной.\n\n#шпера")

            # сохраняем кэш
            save_cache(cache)


# ------- main -------
async def main():
    if not TELEGRAM_BOT_TOKEN or not CHAT_ID:
        logger.error("TELEGRAM_BOT_TOKEN or CHAT_ID not set")
        sys.exit(1)
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    await start_parsing(application)


if __name__ == '__main__':
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
