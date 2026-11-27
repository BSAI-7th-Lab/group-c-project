# Frontend

Leaflet web dashboard for the Margalla Wildfire Planner.

```bash
python -m http.server 8000     # run inside this folder, then open http://localhost:8000
```

| File | Purpose |
|---|---|
| `index.html` | Page layout: map plus control panel |
| `css/style.css` | Styles (to be aligned with the Figma style guide, GC-17) |
| `js/map.js` | Map, topographic tiles, park boundary |
| `data/park_boundary.geojson` | Park outline from the GIS step (GC-10), not yet added |

Next: risk layer, click-to-ignite and fire animation (GC-36); unit markers, ACO routes and charts (GC-37).
