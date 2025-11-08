# Utility: find_best_street_geometry.py
# Finds a good Overpass geometry for a street name near a Nominatim point,
# preferring highway elements and allowing snapping to that geometry.
#
# Usage:
#   from find_best_street_geometry import find_best_street_geometry, nearest_point_on_polyline
#   geom, meta = find_best_street_geometry("Wahlert", "Wuppertal", [51.1998355,7.1096489], snap_max_m=500, debug=True)
#   if geom:
#       pt, dist = nearest_point_on_polyline([lat,lon], geom)
#       # use pt for marker coordinates
import requests
import math


def haversine_m(a, b):
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


def closest_point_on_segment_geo(p, a, b):
    # p,a,b are [lat, lon]
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


def nearest_point_on_polyline(p, coords):
    if not coords:
        return None, float('inf')
    if len(coords) == 1:
        return coords[0], haversine_m(p, coords[0])
    best_pt = None
    best_d = float('inf')
    for i in range(len(coords) - 1):
        a = coords[i]
        b = coords[i + 1]
        c = closest_point_on_segment_geo(p, a, b)
        d = haversine_m(p, c)
        if d < best_d:
            best_d = d
            best_pt = c
    return best_pt, best_d


def overpass_get_element_geometry(osm_type, osm_id, timeout=25):
    if not osm_id:
        return None
    try:
        if osm_type == "way":
            q = f'[out:json][timeout:{timeout}]; way({int(osm_id)}); out geom;'
        else:
            q = f'[out:json][timeout:{timeout}]; relation({int(osm_id)}); out geom;'
        r = requests.post("https://overpass-api.de/api/interpreter",
                          data=q.encode("utf-8"),
                          timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data.get("elements"):
            return None
        el = data["elements"][0]
        geom = el.get("geometry")
        if not geom:
            return None
        return {
            "coords": [[float(p["lat"]), float(p["lon"])] for p in geom],
            "element": el
        }
    except Exception:
        return None


def overpass_search_by_name(street_name, city_name, timeout=25):
    safe_street = street_name.replace('"', '\\"')
    safe_city = city_name.replace('"', '\\"') if city_name else ""
    q = f'''
[out:json][timeout:{timeout}];
area["name"="{safe_city}"]->.a;
(
  way["name"="{safe_street}"](area.a);
  relation["name"="{safe_street}"](area.a);
);
out geom;
'''
    try:
        r = requests.post("https://overpass-api.de/api/interpreter",
                          data=q.encode("utf-8"),
                          timeout=timeout)
        if r.status_code != 200:
            return []
        data = r.json()
        return data.get("elements", [])
    except Exception:
        return []


# simplistic tag priority to prefer roads
_TAG_PRIORITY = {
    "motorway": 100,
    "trunk": 90,
    "primary": 80,
    "secondary": 70,
    "tertiary": 60,
    "residential": 50,
    "unclassified": 40,
    "service": 30,
    "waterway": 5,
    "default": 10
}


def element_priority_score(tags):
    if not tags:
        return _TAG_PRIORITY["default"]
    hw = tags.get("highway")
    if hw:
        return _TAG_PRIORITY.get(hw, _TAG_PRIORITY["default"]) + 1000
    if "waterway" in tags:
        return _TAG_PRIORITY.get("waterway", _TAG_PRIORITY["default"])
    return _TAG_PRIORITY["default"]


def pick_best_candidate_from_overpass(elements,
                                      nom_point,
                                      prefer_highway=True,
                                      debug=False):
    best = None
    best_score = -1e12
    for el in elements:
        tags = el.get("tags", {}) or {}
        geom = el.get("geometry") or []
        if not geom:
            continue
        coords = [[float(p["lat"]), float(p["lon"])] for p in geom]
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        centroid = (sum(lats) / len(lats), sum(lons) / len(lons))
        dist = haversine_m(nom_point, centroid)
        pri = element_priority_score(tags)
        score = pri * 1000.0 - dist
        if prefer_highway and tags.get("highway"):
            score += 500000
        if debug:
            print(" candidate id", el.get("id"), "tags sample:",
                  {k: tags.get(k)
                   for k in list(tags)[:4]}, "centroid_dist_m:", int(dist),
                  "score:", int(score))
        if score > best_score:
            best_score = score
            best = {
                "element": el,
                "coords": coords,
                "centroid": centroid,
                "dist_to_nom": dist,
                "score": score
            }
    return best


def find_best_street_geometry(street,
                              city,
                              nom_point,
                              snap_max_m=500,
                              prefer_highway=True,
                              debug=False,
                              try_element_first=True,
                              osm_type=None,
                              osm_id=None):
    """
    street: street name string
    city: city name
    nom_point: [lat, lon] from Nominatim
    snap_max_m: max allowed nearest-point distance (meters) to accept geometry
    prefer_highway: prefer highway-tagged elements over waterway/stop_area
    try_element_first: if osm_type/osm_id provided, attempt fetching that element first
    osm_type/osm_id: optional, pass the loc.raw['osm_type'] and ['osm_id'] if available
    Returns: (coords_list, meta) or (None, None)
    """
    if debug:
        print("find_best_street_geometry:", street, city, "nom_point:",
              nom_point)
    # 1) try direct element fetch if caller provided osm_type/osm_id and try_element_first True
    if try_element_first and osm_type and osm_id:
        if debug:
            print("trying element fetch for", osm_type, osm_id)
        el = overpass_get_element_geometry(osm_type, osm_id)
        if el and el.get("coords"):
            # if element has highway tag prefer it; otherwise still consider, but check distance
            tags = el.get("element", {}).get("tags", {}) or {}
            centroid = (sum([p[0] for p in el["coords"]]) / len(el["coords"]),
                        sum([p[1] for p in el["coords"]]) / len(el["coords"]))
            d_cent = haversine_m(nom_point, centroid)
            pt, d_near = nearest_point_on_polyline(nom_point, el["coords"])
            if debug:
                print(" element fetched: centroid_dist", int(d_cent),
                      "nearest_point_dist", int(d_near))
            if d_near <= snap_max_m:
                meta = {
                    "element": el.get("element"),
                    "centroid": centroid,
                    "centroid_dist_m": d_cent,
                    "nearest_point_dist_m": d_near,
                    "reason": "element_by_osm_id"
                }
                return el["coords"], meta
            # else continue to name search
    # 2) search by name and pick best candidate
    elems = overpass_search_by_name(street, city)
    if debug:
        print("Overpass returned", len(elems), "elements for name", street)
    if not elems:
        return None, None
    best = pick_best_candidate_from_overpass(elems,
                                             nom_point,
                                             prefer_highway=prefer_highway,
                                             debug=debug)
    if not best:
        return None, None
    pt, d_near = nearest_point_on_polyline(nom_point, best["coords"])
    if debug:
        print("best candidate centroid_dist:", int(best["dist_to_nom"]),
              "nearest_pt_dist:", int(d_near))
    if d_near <= snap_max_m:
        meta = {
            "element": best["element"],
            "centroid": best["centroid"],
            "centroid_dist_m": best["dist_to_nom"],
            "nearest_point_dist_m": d_near,
            "score": best["score"],
            "reason": "name_search"
        }
        return best["coords"], meta
    if debug:
        print("No candidate within snap_max_m:", snap_max_m)
    return None, None


# expose nearest_point_on_polyline for callers
__all__ = ["find_best_street_geometry", "nearest_point_on_polyline"]
