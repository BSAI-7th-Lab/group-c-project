// Dashboard skeleton (Jira GC-15).
// Map of Margalla Hills with topographic tiles and the park boundary.
// The boundary file comes from the data preparation step (Khubaib, GC-10):
// save it as frontend/data/park_boundary.geojson.

const MARGALLA_CENTRE = [33.755, 73.06];   // approximate centre of the national park
const BOUNDARY_URL = "data/park_boundary.geojson";

const map = L.map("map", { zoomControl: true }).setView(MARGALLA_CENTRE, 12);

L.tileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", {
  maxZoom: 17,
  attribution:
    'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, ' +
    'SRTM | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (CC-BY-SA)',
}).addTo(map);

L.control.scale({ imperial: false }).addTo(map);

function showNotice(text) {
  const Notice = L.Control.extend({
    onAdd() {
      const div = L.DomUtil.create("div", "map-notice");
      div.textContent = text;
      return div;
    },
  });
  new Notice({ position: "bottomleft" }).addTo(map);
}

fetch(BOUNDARY_URL)
  .then((res) => {
    if (!res.ok) throw new Error(res.status);
    return res.json();
  })
  .then((geojson) => {
    const boundary = L.geoJSON(geojson, {
      style: { color: "#2f6b45", weight: 2, fillOpacity: 0.05 },
    }).addTo(map);
    map.fitBounds(boundary.getBounds(), { padding: [20, 20] });
  })
  .catch(() => showNotice("Park boundary not loaded yet (add frontend/data/park_boundary.geojson)"));

// Live labels for the controls
for (const [inputId, outId] of [["wind-speed", "wind-speed-out"], ["wind-dir", "wind-dir-out"]]) {
  const input = document.getElementById(inputId);
  const out = document.getElementById(outId);
  input.addEventListener("input", () => { out.textContent = input.value; });
}
