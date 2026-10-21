# Third-party notices

## Dashboard design references

The UI was redesigned using the data hierarchy, restrained component styling, and responsive patterns found in the MIT-licensed [ObservedObserver/streamlit-shadcn-ui](https://github.com/ObservedObserver/streamlit-shadcn-ui) and [shadcn-ui/ui](https://github.com/shadcn-ui/ui) projects, alongside the [official Streamlit theme guidance](https://github.com/streamlit/streamlit/blob/develop/lib/streamlit/.agents/skills/developing-with-streamlit/references/theme.md). No source code or CSS was copied; `legacy_streamlit/static/dashboard.css` is project-authored and keeps MaydonOs' cream and burgundy palette.

## Noto Sans

The interface includes the Noto Sans variable webfont (`legacy_streamlit/static/fonts/NotoSans-Variable.woff2`, copied to `frontend/assets/` for the public web app). Noto Sans is distributed under the SIL Open Font License, Version 1.1; the license text is included at `legacy_streamlit/static/fonts/OFL.txt`. Upstream: [Google Fonts Noto Sans](https://github.com/google/fonts/tree/main/ofl/notosans).

## Leaflet and Esri basemap

The public web map loads Leaflet 1.9.4 from its public CDN and uses Esri World Imagery tiles. The map includes Esri's imagery attribution; basemap availability and use are subject to Esri's service terms.

## MIT-licensed source component

This project incorporates a source component distributed under the MIT License. The required copyright and permission notice is retained below.

```text
MIT License

Copyright (c) 2026 Krishna Sharma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## OpenStreetMap

OpenStreetMap data is © OpenStreetMap contributors and is available under the Open Database License (ODbL). The MaydonOs interface attributes the map context and does not treat OSM tags as ground truth.

## Copernicus CORINE Land Cover

The application queries and displays CORINE Land Cover 2018 through the Copernicus Land Monitoring Service / European Environment Agency ArcGIS services. © European Environment Agency. CORINE classes are regional land-cover context; the 2018 status layer has a 25 ha minimum mapping unit and must not be interpreted as parcel-level ground truth.
