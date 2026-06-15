"""Headless Chrome SVG render benchmarking.

Measures actual browser DOM parse + layout time for SVG content.
Uses Playwright with Chromium for accurate results.

Usage:
    from svgym.render_bench import RenderBenchmark

    async with RenderBenchmark() as bench:
        ms = await bench.measure(svg_text)
        results = await bench.compare(raw_svg, optimized_svg)
"""

import json
import asyncio
from contextlib import asynccontextmanager

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head><style>body { margin: 0; } #container { width: 512px; height: 512px; }</style></head>
<body>
<div id="container"></div>
<script>
async function measureRender(svgText, iterations) {
    const container = document.getElementById('container');

    // Warmup
    container.innerHTML = svgText;
    container.offsetHeight;
    container.innerHTML = '';

    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) {
        container.innerHTML = svgText;
        container.offsetHeight;
        getComputedStyle(container).opacity;
        container.innerHTML = '';
    }
    const t1 = performance.now();

    // Return total time for all iterations
    return t1 - t0;
}

window.measureRender = measureRender;
</script>
</body>
</html>"""


class RenderBenchmark:
    """Headless Chrome SVG render benchmarker."""

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._page = None

    async def start(self):
        from playwright.async_api import async_playwright
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        self._page = await self._browser.new_page()
        await self._page.set_content(HTML_TEMPLATE)
        await self._page.wait_for_function("window.measureRender")

    async def stop(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()

    async def measure(self, svg_text: str, iterations: int = 30) -> float:
        """Measure total time to render SVG `iterations` times in ms."""
        escaped = json.dumps(svg_text)
        result = await self._page.evaluate(
            f"measureRender({escaped}, {iterations})"
        )
        return result

    async def compare(self, raw_svg: str, optimized_svg: str,
                      svgo_svg: str = None, iterations: int = 30) -> dict:
        """Compare render times between raw, optimized, and optionally SVGO.

        Returns total time for `iterations` renders of each variant.
        Speedup = total_raw / total_optimized.
        """
        raw_ms = await self.measure(raw_svg, iterations)
        opt_ms = await self.measure(optimized_svg, iterations)

        result = {
            "raw_ms": round(raw_ms, 3),
            "optimized_ms": round(opt_ms, 3),
            "speedup": round(raw_ms / opt_ms, 2) if opt_ms > 0 else 0,
        }

        if svgo_svg is not None:
            svgo_ms = await self.measure(svgo_svg, iterations)
            result["svgo_ms"] = round(svgo_ms, 3)
            result["speedup_vs_svgo"] = round(svgo_ms / opt_ms, 2) if opt_ms > 0 else 0

        return result
