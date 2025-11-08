"""
diagnose_street.py

Lightweight diagnostic tool to inspect Nominatim + Overpass results for a given street.
Use it to reproduce the behavior for "Wahlert, Wuppertal" (or any other street+city).

Dependencies:
    pip install geopy requests folium

Example usage:
    python diagnose_street.py

It will:
 - try several geocoding queries for the street,
 - print Nominatim raw results (lat/lon, osm_type/id, address fields, bbox),
 - if osm_type is way/relation and osm_id present, fetch that element geometry from Overpass and print centroid/distance,
 - run an Overpass name search in the given city area and list matching elements and a small sample of coords,
 - compute nearest-point distances between Nominatim point and found geometries,
 - optionally build a small folium map ('diagnose_<street>_<city>.html') to visualize results (toggle with SAVE_MAP).
"""
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import requests, json, time, math, os, sys
from pprint import pprint

SAVE_MAP = True  # set to False to avoid creating folium map file
MAP_FILENAME_TEMPLATE = "diagnose_{street}_{city}.html"


def haversine_m(a, b):
    # a,b = (lat, lon) in degrees
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
    # osm_type: 'way' or 'relation' or 'node' (we only handle way/relation geometry)
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
            print("Overpass element fetch failed (status):", r.status_code)
            return None
        data = r.json()
        if not data.get("elements"):
            print("No elements returned for element query")
            return None
        el = data["elements"][0]
        geom = el.get("geometry")
        if not geom:
            return None
        coords = [[float(p["lat"]), float(p["lon"])] for p in geom]
        return coords
    except Exception as e:
        print("Overpass element fetch exception:", e)
        return None


def overpass_search_by_name(street_name, city_name, timeout=25, limit=50):
    # search ways/relations with name==street_name within area[city_name]
    safe_street = street_name.replace('"', '\\"')
    safe_city = city_name.replace('"', '\\"') if city_name else ""
    q = f'''
[out:json][timeout:{timeout}];
area["name"="{safe_city}"]->.a;
(
  way["name"="{safe_street}"](area.a);
  relation["name"="{safe_street}"](area.a);
);
out geom {limit};
'''
    try:
        r = requests.post("https://overpass-api.de/api/interpreter",
                          data=q.encode("utf-8"),
                          timeout=timeout)
        if r.status_code != 200:
            print("Overpass search failed (status):", r.status_code)
            try:
                print(r.text[:1000])
            except Exception:
                pass
            return []
        data = r.json()
        return data.get("elements", [])
    except Exception as e:
        print("Overpass search exception:", e)
        return []


def run_diagnostic(street,
                   city,
                   snap_threshold_m=500,
                   save_map=SAVE_MAP,
                   debug=True):
    geolocator = Nominatim(user_agent="diagnose_street_tool")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    queries = [
        f"{street}, {city}" if city else street,
        f"{street}straße, {city}" if city else f"{street}straße",
        f"{street}str., {city}" if city else f"{street}str.", street
    ]

    print("=" * 80)
    print("Diagnostic for:", street, ",", city)
    print("Will try queries:", queries)
    print("=" * 80)

    geocoded = []
    for q in queries:
        try:
            print("\nQuery:", q)
            loc = geocode(q, addressdetails=True)
        except Exception as e:
            print("Geocode exception for query:", q, "->", e)
            loc = None
        if not loc:
            print("  -> no result")
            continue
        raw = loc.raw or {}
        addr = raw.get("address", {}) or {}
        print("  lat,lon:", loc.latitude, loc.longitude)
        print("  display_name:", raw.get("display_name"))
        print("  osm_type:", raw.get("osm_type"), "osm_id:", raw.get("osm_id"),
              "type:", raw.get("type"))
        sample = {
            k: addr.get(k)
            for k in ("road", "house_number", "city_district", "suburb",
                      "neighbourhood", "village", "county", "state",
                      "postcode", "country")
        }
        print("  address fields:", sample)
        print("  boundingbox:", raw.get("boundingbox"))
        geocoded.append({"query": q, "loc": loc, "raw": raw})
        time.sleep(1.1)

    if not geocoded:
        print("\nNo geocoding results found for any query. Stopping.")
        return

    # pick the first successful result for further inspection (you can change logic)
    chosen = geocoded[0]
    nom_lat = float(chosen["loc"].latitude)
    nom_lon = float(chosen["loc"].longitude)
    nom_point = [nom_lat, nom_lon]
    nom_raw = chosen["raw"]
    osm_type = (nom_raw.get("osm_type") or "").lower()
    osm_id = nom_raw.get("osm_id")

    print("\n" + "=" * 60)
    print("Chosen Nominatim result (first success):")
    print(" point:", nom_point)
    print(" osm_type:", osm_type, "osm_id:", osm_id)
    print(" display_name:", nom_raw.get("display_name"))
    print("=" * 60)

    # If osm_type is way/relation with osm_id -> fetch geometry for that element
    element_geom = None
    if osm_type in ("way", "relation") and osm_id:
        print("\nFetching geometry for reported osm element (", osm_type,
              osm_id, ") via Overpass...")
        element_geom = overpass_get_element_geometry(osm_type, osm_id)
        if element_geom:
            # compute centroid
            lats = [p[0] for p in element_geom]
            lons = [p[1] for p in element_geom]
            centroid = (sum(lats) / len(lats), sum(lons) / len(lons))
            d_c = int(haversine_m(nom_point, centroid))
            print(" element geometry length:", len(element_geom))
            print(" centroid:", centroid, "distance nominatim->centroid (m):",
                  d_c)
            print(" sample coords (first 6):", element_geom[:6])
        else:
            print(" No geometry returned for that element.")

    # Overpass search by name within city (list candidates)
    print("\nSearching Overpass for elements named", street, "within", city,
          "...")
    elems = overpass_search_by_name(street, city)
    print(" Overpass returned", len(elems), "elements.")
    # For each element print brief info
    candidates = []
    for el in elems:
        etype = el.get("type")
        eid = el.get("id")
        has_geom = bool(el.get("geometry"))
        name = el.get("tags", {}).get("name")
        tags = el.get("tags", {})
        print(
            f" - element id={eid} type={etype} name={name} has_geom={has_geom} tags_sample={ {k:tags[k] for k in list(tags)[:4]} }"
        )
        geom_coords = []
        if has_geom:
            geom_coords = [[float(p["lat"]), float(p["lon"])]
                           for p in el["geometry"]]
            # compute centroid
            lats = [p[0] for p in geom_coords]
            lons = [p[1] for p in geom_coords]
            centroid = (sum(lats) / len(lats), sum(lons) / len(lons))
            dist_to_nom = int(haversine_m(nom_point, centroid))
            print("   centroid:", centroid,
                  "distance nominatim->centroid (m):", dist_to_nom,
                  "geom_len:", len(geom_coords))
        candidates.append({"el": el, "geom": geom_coords})

    # For each candidate geometry compute nearest point from nominatim point and print distance
    for idx, c in enumerate(candidates):
        coords = c.get("geom")
        if not coords:
            continue
        pt, d = nearest_point_on_polyline(nom_point, coords)
        print(
            f"\n Candidate #{idx} nearest distance to nominatim point: {int(d)} m; nearest_pt={pt}"
        )
        if debug:
            # find where along the geometry the nearest segment was (approx): show sample around nearest index
            # naive search for nearest vertex index
            best_i = None
            best_v = float('inf')
            for i, v in enumerate(coords):
                dv = haversine_m(nom_point, v)
                if dv < best_v:
                    best_v = dv
                    best_i = i
            sample_range = coords[max(0, best_i -
                                      3):min(len(coords), best_i + 4)]
            print("   approx nearest vertex idx:", best_i, "vertex_dist:",
                  int(best_v), "sample around vertex:", sample_range)

    # Optionally build a small folium map to visualize nominatim point, element geometry and candidates
    if save_map:
        try:
            import folium
            map_center = nom_point
            fmap = folium.Map(location=map_center, zoom_start=15)
            # nominatim point marker (purple)
            folium.CircleMarker(location=nom_point,
                                radius=6,
                                color="purple",
                                fill=True,
                                fill_color="purple",
                                popup="Nominatim point").add_to(fmap)
            # element geometry (blue)
            if element_geom:
                folium.PolyLine(element_geom,
                                color="#3388ff",
                                weight=4,
                                opacity=0.6,
                                popup=f"{osm_type}/{osm_id}").add_to(fmap)
            # candidate geometries (green)
            for i, c in enumerate(candidates):
                coords = c.get("geom")
                if coords:
                    folium.PolyLine(coords,
                                    color="#13a10e",
                                    weight=3,
                                    opacity=0.5,
                                    popup=f"candidate {i}").add_to(fmap)
            # add markers for candidate centroids
            for i, c in enumerate(candidates):
                coords = c.get("geom")
                if coords:
                    lats = [p[0] for p in coords]
                    lons = [p[1] for p in coords]
                    centroid = (sum(lats) / len(lats), sum(lons) / len(lons))
                    folium.CircleMarker(
                        location=centroid,
                        radius=4,
                        color="green",
                        fill=True,
                        fill_color="green",
                        popup=f"cand {i} centroid").add_to(fmap)
            fn = MAP_FILENAME_TEMPLATE.format(street=street.replace(" ", "_"),
                                              city=city.replace(" ", "_"))
            fmap.save(fn)
            print("\nMap saved to", fn,
                  "- open in browser to inspect visually.")
        except Exception as e:
            print("Failed to create folium map:", e)

    print("\nDiagnostic finished.")
    return {
        "nominatim_chosen": chosen,
        "element_geom": element_geom,
        "overpass_candidates": candidates
    }


if __name__ == "__main__":
    # default test for Wahlert, Wuppertal
    street = "Wahlert"
    city = "Wuppertal"
    out = run_diagnostic(street,
                         city,
                         snap_threshold_m=500,
                         save_map=SAVE_MAP,
                         debug=True)
    # print a compact summary
    print("\nSUMMARY:")
    if out.get("nominatim_chosen"):
        ch = out["nominatim_chosen"]
        print(" Nominatim chosen query:", ch["query"])
        raw = ch["raw"]
        print("  osm_type:", raw.get("osm_type"), "osm_id:", raw.get("osm_id"))
        print("  display_name:", raw.get("display_name"))
    print(" Overpass candidate count:",
          len(out.get("overpass_candidates") or []))
