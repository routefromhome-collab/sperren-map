# debug_place_check.py
import re, time, math, requests, unicodedata, logging
from pprint import pprint
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


def _haversine_m(a, b):
    R = 6371000.0
    φ1 = math.radians(a[0])
    λ1 = math.radians(a[1])
    φ2 = math.radians(b[0])
    λ2 = math.radians(b[1])
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    sa = math.sin(
        dφ / 2.0)**2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(sa), math.sqrt(1.0 - sa))
    return R * c


def fetch_osm_geom(osm_type, osm_id, timeout=20):
    try:
        q = f'[out:json][timeout:{timeout}]; {osm_type}({int(osm_id)}); out geom;'
        r = requests.post("https://overpass-api.de/api/interpreter",
                          data=q.encode("utf-8"),
                          timeout=timeout)
        if r.status_code != 200:
            return None, f"overpass_status_{r.status_code}"
        j = r.json()
        elems = j.get("elements", [])
        if not elems:
            return None, "no_elems"
        el = elems[0]
        geom = []
        if el.get("geometry"):
            for p in el["geometry"]:
                geom.append([float(p["lat"]), float(p["lon"])])
        else:
            if el.get("lat") and el.get("lon"):
                geom = [[float(el["lat"]), float(el["lon"])]]
            else:
                c = el.get("center") or {}
                if c and "lat" in c and "lon" in c:
                    geom = [[float(c["lat"]), float(c["lon"])]]
        return geom, None
    except Exception as e:
        return None, f"exception:{e}"


def nearest_point_on_segment_geo(p, a, b):
    latScale = math.cos(math.radians(p[0]))
    ax, ay = a[0], a[1] * latScale
    bx, by = b[0], b[1] * latScale
    px, py = p[0], p[1] * latScale
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vlen2 = vx * vx + vy * vy
    if vlen2 == 0:
        return [a[0], a[1]]
    t = (vx * wx + vy * wy) / vlen2
    t = max(0.0, min(1.0, t))
    cx = ax + vx * t
    cy = ay + vy * t
    return [cx, cy / latScale]


def nearest_point_on_polygon(p, poly_coords):
    best = None
    best_d = float('inf')
    n = len(poly_coords)
    if n == 0:
        return None, float('inf')
    if n == 1:
        return poly_coords[0], _haversine_m(p, poly_coords[0])
    for i in range(n):
        a = poly_coords[i]
        b = poly_coords[(i + 1) % n]
        c = nearest_point_on_segment_geo(p, a, b)
        d = _haversine_m(p, c)
        if d < best_d:
            best_d = d
            best = c
    return best, best_d


def debug_check_addresses(addresses,
                          city="Wuppertal",
                          pause=1.0,
                          proximity_thresh=4.0):
    geolocator = Nominatim(user_agent="debug_place_check")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=pause)
    seen_coords = []
    for addr in addresses:
        print("\n=== Address:", addr)
        try:
            q = f"{addr}, {city}" if city else addr
            loc = geocode(q, addressdetails=True)
        except Exception as e:
            print("Geocode failed:", e)
            loc = None
        if not loc:
            print("Result: nominatim returned NOTHING")
            continue
        raw = loc.raw or {}
        af = raw.get("address", {}) or {}
        print("Nominatim: lat,lon =", loc.latitude, loc.longitude, "osm_type=",
              raw.get("osm_type"), "osm_id=", raw.get("osm_id"))
        pprint(af)
        # If way/relation, try fetch geom and calc nearest perimeter
        osm_type = (raw.get("osm_type") or "").lower()
        osm_id = raw.get("osm_id")
        chosen_use = None
        refined_from_building = False
        if osm_type in ("way", "relation") and osm_id:
            geom, err = fetch_osm_geom(osm_type, osm_id)
            if err:
                print("Overpass fetch error:", err)
            else:
                print("Fetched geometry points:", len(geom))
                nom_point = [float(loc.latitude), float(loc.longitude)]
                per_pt, per_d = nearest_point_on_polygon(nom_point, geom)
                centroid = (sum([p[0] for p in geom]) / len(geom),
                            sum([p[1]
                                 for p in geom]) / len(geom)) if geom else None
                print(
                    "Centroid:", centroid, "dist nom->centroid (m):",
                    int(_haversine_m(nom_point, centroid))
                    if centroid else None)
                print("Nearest perimeter pt:", per_pt, "dist m:",
                      int(per_d) if per_d != float('inf') else None)
                if per_pt:
                    chosen_use = per_pt
                    refined_from_building = True
        # Check whether we'd skip due to proximity (simulate)
        if chosen_use:
            use = chosen_use
        else:
            use = [float(loc.latitude), float(loc.longitude)]
        too_close = False
        for (ex_lat, ex_lon) in seen_coords:
            d = _haversine_m(use, [ex_lat, ex_lon])
            if d <= proximity_thresh:
                too_close = True
                print(
                    f"Would SKIP due to proximity: dist {d:.1f} m to existing marker at {ex_lat},{ex_lon}"
                )
                break
        if not too_close:
            print("Would ADD marker at", use, "refined_from_building=",
                  refined_from_building)
            seen_coords.append((use[0], use[1]))
    print("\nDebug run complete.")


if __name__ == "__main__":
    test_addrs = [
        "Katernberger Str 282", "Katernberger Str 292", "Katernberger Str 250"
        # добавьте примеры, которые у вас сейчас НЕ ставятся
    ]
    debug_check_addresses(test_addrs,
                          city="Wuppertal",
                          pause=1.0,
                          proximity_thresh=4.0)
